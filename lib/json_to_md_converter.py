#!/usr/bin/env python3
"""
Schema-Driven JSON to Markdown Renderer
========================================
Automatically converts EIMAS JSON output to Markdown without hardcoding.

Architecture:
1. normalize(raw_json) -> ReportModel (standard structure)
2. render_md(ReportModel) -> Markdown

Features:
- Section-driven: renders only what exists in JSON
- Auto-generates tables from list-of-objects
- Handles unknown fields in extra_fields section
- Validates output against input
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


# ============================================================================
# SECTION SCHEMA DEFINITION
# ============================================================================

SECTION_SCHEMA = {
    # Priority determines rendering order (lower = first)
    "decision_policy": {"priority": 1, "title": "💡 의사결정 정책", "icon": "💡"},
    "score_definitions": {"priority": 2, "title": "📊 리스크 점수 정의", "icon": "📊"},
    "allocation": {"priority": 3, "title": "📈 포트폴리오 배분", "icon": "📈"},
    "portfolio_weights": {"priority": 3.5, "title": "⚖️ 포트폴리오 비중", "icon": "⚖️"},
    "constraint_repair": {"priority": 4, "title": "🔧 제약조건 수정", "icon": "🔧"},
    "rebalance_plan": {"priority": 5, "title": "⚖️ 리밸런싱 계획", "icon": "⚖️"},
    "hold_policy": {"priority": 6, "title": "🛑 HOLD 정책", "icon": "🛑"},
    "signal_hierarchy": {"priority": 7, "title": "📡 시그널 계층", "icon": "📡"},
    "integrated_signals": {"priority": 8, "title": "🔗 통합 시그널", "icon": "🔗"},
    "fomc_analysis": {"priority": 10, "title": "🏦 FOMC 분석", "icon": "🏦"},
    "fred_summary": {"priority": 11, "title": "🏛️ 거시경제 지표", "icon": "🏛️"},
    "regime": {"priority": 12, "title": "📈 시장 레짐", "icon": "📈"},
    "debate_consensus": {"priority": 13, "title": "🤖 AI 토론 합의", "icon": "🤖"},
    "debate_results": {"priority": 13.5, "title": "💬 토론 결과", "icon": "💬"},
    "validation_loop_result": {"priority": 14, "title": "✅ 검증 루프", "icon": "✅"},
    "verification": {"priority": 15, "title": "🔍 검증 결과", "icon": "🔍"},
    "market_quality": {"priority": 16, "title": "🎯 시장 품질", "icon": "🎯"},
    "bubble_risk": {"priority": 17, "title": "💥 버블 리스크", "icon": "💥"},
    "genius_act_regime": {"priority": 18, "title": "💧 Genius Act 레짐", "icon": "💧"},
    "genius_act_signals": {"priority": 18.5, "title": "💧 Genius Act 시그널", "icon": "💧"},
    "theme_etf_analysis": {"priority": 19, "title": "🎨 테마 ETF", "icon": "🎨"},
    "etf_flow_result": {"priority": 19.5, "title": "📊 ETF 플로우", "icon": "📊"},
    "shock_propagation": {"priority": 20, "title": "🌊 충격 전파", "icon": "🌊"},
    "ark_analysis": {"priority": 21, "title": "🚀 ARK Invest", "icon": "🚀"},
    "sentiment_analysis": {"priority": 22, "title": "😊 센티먼트", "icon": "😊"},
    "extended_data": {"priority": 22.5, "title": "📊 확장 데이터", "icon": "📊"},
    "events_detected": {"priority": 23, "title": "📅 이벤트 탐지", "icon": "📅"},
    "event_tracking": {"priority": 23.5, "title": "📅 이벤트 추적", "icon": "📅"},
    "tracked_events": {"priority": 23.6, "title": "📅 추적 중 이벤트", "icon": "📅"},
    "event_predictions": {"priority": 23.7, "title": "🔮 이벤트 예측", "icon": "🔮"},
    "event_attributions": {"priority": 23.8, "title": "🎯 이벤트 귀인", "icon": "🎯"},
    "event_backtest_results": {"priority": 23.9, "title": "📈 이벤트 백테스트", "icon": "📈"},
    "volume_anomalies": {"priority": 24, "title": "📊 거래량 이상", "icon": "📊"},
    "critical_path_monitoring": {"priority": 25, "title": "🛤️ 크리티컬 패스", "icon": "🛤️"},
    "correlation_matrix": {"priority": 26, "title": "🔗 상관관계 행렬", "icon": "🔗"},
    "correlation_tickers": {"priority": 26.5, "title": "📊 상관관계 티커", "icon": "📊"},
    "adaptive_portfolios": {"priority": 27, "title": "🎯 적응형 포트폴리오", "icon": "🎯"},
    "crypto_monitoring": {"priority": 28, "title": "₿ 크립토 모니터링", "icon": "₿"},
    "crypto_stress_test": {"priority": 28.5, "title": "⚠️ 크립토 스트레스", "icon": "⚠️"},
    "defi_tvl": {"priority": 29, "title": "🏦 DeFi TVL", "icon": "🏦"},
    "onchain_risk_signals": {"priority": 29.5, "title": "⛓️ 온체인 리스크", "icon": "⛓️"},
    "mena_markets": {"priority": 30, "title": "🌍 MENA 시장", "icon": "🌍"},
    "intraday_summary": {"priority": 31, "title": "⏰ 일중 요약", "icon": "⏰"},
    "news_correlations": {"priority": 32, "title": "📰 뉴스 상관관계", "icon": "📰"},
    "ai_report": {"priority": 40, "title": "🤖 AI 리포트", "icon": "🤖"},
    "agent_outputs": {"priority": 41, "title": "🤖 에이전트 출력", "icon": "🤖"},
    "reasoning_chain": {"priority": 42, "title": "🧠 추론 체인", "icon": "🧠"},
    "devils_advocate_arguments": {"priority": 43, "title": "😈 반대 논거", "icon": "😈"},
    "dissent_records": {"priority": 44, "title": "⚠️ 이견 기록", "icon": "⚠️"},
    "trade_plan": {"priority": 50, "title": "📝 거래 계획", "icon": "📝"},
    "audit_metadata": {"priority": 99, "title": "📋 감사 메타데이터", "icon": "📋"},
}

# Table column ordering preferences
TABLE_COLUMN_ORDER = [
    "ticker", "asset", "asset_class",
    "current_weight", "target_weight", "delta_weight", "delta_pct",
    "action", "priority", "estimated_cost",
    "value", "signal", "regime", "type",
    "source", "confidence", "timestamp"
]


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class ReportModel:
    """Normalized report structure"""
    schema_version: str
    timestamp: str

    # Core sections (ordered by priority)
    sections: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    final_recommendation: Optional[str] = None
    confidence: Optional[float] = None
    risk_score: Optional[float] = None

    # Extra fields not in schema
    extra_fields: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# NORMALIZATION LAYER
# ============================================================================

def normalize(raw_json: Dict[str, Any]) -> ReportModel:
    """
    Convert raw JSON to normalized ReportModel

    Strategy:
    1. Extract known sections from SECTION_SCHEMA
    2. Store everything else in extra_fields
    3. Preserve metadata for validation
    """
    schema_version = raw_json.get('schema_version', '1.0.0')
    timestamp = raw_json.get('timestamp', datetime.now().isoformat())

    sections = {}
    extra_fields = {}

    # Metadata fields (handled separately, not in extra)
    metadata_fields = {
        'timestamp', 'schema_version', 'final_recommendation',
        'confidence', 'risk_score', 'risk_level',
        'market_data_count', 'crypto_data_count',
        'liquidity_signal', 'has_strong_dissent',
        'full_mode_position', 'reference_mode_position', 'modes_agree',
        'warnings', 'realtime_signals',
        'whitening_summary', 'fact_check_grade',
        'base_risk_score', 'microstructure_adjustment',
        'bubble_risk_adjustment', 'extended_data_adjustment',
        'hrp_allocation_rationale', 'volume_analysis_summary'
    }

    # Extract known sections
    for key, value in raw_json.items():
        if key in SECTION_SCHEMA:
            sections[key] = value
        elif key in metadata_fields:
            # Metadata - handled separately
            continue
        else:
            # Unknown field - will be rendered in extra section
            extra_fields[key] = value

    # Handle nested operational_report
    if 'operational_report' in raw_json:
        op_report = raw_json['operational_report']
        if isinstance(op_report, dict):
            for key, value in op_report.items():
                if key in SECTION_SCHEMA:
                    sections[key] = value
                else:
                    extra_fields[f'operational_report.{key}'] = value

    return ReportModel(
        schema_version=schema_version,
        timestamp=timestamp,
        sections=sections,
        final_recommendation=raw_json.get('final_recommendation'),
        confidence=raw_json.get('confidence'),
        risk_score=raw_json.get('risk_score'),
        extra_fields=extra_fields
    )


# ============================================================================
# RENDERING UTILITIES
# ============================================================================

def format_value(value: Any, key: str = "") -> str:
    """Format a value for markdown display"""
    if value is None:
        return "N/A"
    elif isinstance(value, bool):
        return "예" if value else "아니오"
    elif isinstance(value, float):
        # Smart formatting based on key name
        if 'weight' in key or 'ratio' in key or 'confidence' in key:
            return f"{value * 100:.2f}%" if value <= 1 else f"{value:.2f}"
        elif 'score' in key or 'risk' in key:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"
    elif isinstance(value, (int)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        return f"{len(value)} items"
    elif isinstance(value, dict):
        return f"{len(value)} fields"
    else:
        return str(value)


def auto_table(data: List[Dict[str, Any]], max_rows: int = 20) -> str:
    """
    Auto-generate markdown table from list of objects

    Strategy:
    1. Collect all unique keys
    2. Order by TABLE_COLUMN_ORDER preference
    3. Generate table with max_rows limit
    """
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return ""

    # Collect all unique keys
    all_keys = set()
    for item in data:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    # Order keys by preference
    ordered_keys = []
    for pref_key in TABLE_COLUMN_ORDER:
        if pref_key in all_keys:
            ordered_keys.append(pref_key)
            all_keys.remove(pref_key)
    # Add remaining keys
    ordered_keys.extend(sorted(all_keys))

    # Build table
    lines = []

    # Header
    header = "| " + " | ".join(ordered_keys) + " |"
    separator = "|" + "|".join(["---" for _ in ordered_keys]) + "|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for item in data[:max_rows]:
        if isinstance(item, dict):
            row_values = [format_value(item.get(k), k) for k in ordered_keys]
            row = "| " + " | ".join(row_values) + " |"
            lines.append(row)

    if len(data) > max_rows:
        lines.append(f"| ... | ({len(data) - max_rows}개 생략) |")

    return "\n".join(lines)


def render_value(value: Any, key: str = "", indent: int = 0) -> List[str]:
    """
    Recursively render a value to markdown lines

    Strategy:
    - dict -> nested list
    - list of dicts -> table
    - list of primitives -> bullet list
    - primitive -> formatted value
    """
    prefix = "  " * indent
    lines = []

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}- **{k}:**")
                lines.extend(render_value(v, k, indent + 1))
            else:
                formatted = format_value(v, k)
                lines.append(f"{prefix}- **{k}:** {formatted}")

    elif isinstance(value, list):
        if not value:
            return [f"{prefix}(empty)"]

        # Check if list of dicts -> table
        if all(isinstance(item, dict) for item in value):
            table = auto_table(value)
            if table:
                lines.append(table)
            else:
                # Fallback to nested rendering
                for i, item in enumerate(value[:10]):
                    lines.append(f"{prefix}- Item {i+1}:")
                    lines.extend(render_value(item, "", indent + 1))
                if len(value) > 10:
                    lines.append(f"{prefix}  ... ({len(value) - 10}개 생략)")
        else:
            # List of primitives
            for item in value[:20]:
                lines.append(f"{prefix}- {format_value(item, key)}")
            if len(value) > 20:
                lines.append(f"{prefix}  ... ({len(value) - 20}개 생략)")

    else:
        lines.append(f"{prefix}{format_value(value, key)}")

    return lines


def render_section(section_key: str, section_data: Any) -> str:
    """Render a single section to markdown"""
    schema = SECTION_SCHEMA.get(section_key, {})
    title = schema.get('title', section_key)

    lines = [f"## {title}", ""]

    # Render content
    content_lines = render_value(section_data, section_key)
    lines.extend(content_lines)

    return "\n".join(lines)


# ============================================================================
# MAIN RENDERER
# ============================================================================

def render_md(model: ReportModel) -> str:
    """
    Render ReportModel to Markdown

    Strategy:
    1. Header with metadata
    2. Executive summary
    3. Sections in priority order
    4. Extra fields
    5. Footer with validation
    """
    parts = []

    # ========== HEADER ==========
    ts = model.timestamp[:19].replace('T', ' ')
    parts.append(f"# 📊 EIMAS 분석 리포트")
    parts.append(f"\n**생성 시간:** {ts}")
    parts.append(f"**스키마 버전:** {model.schema_version}")

    # ========== EXECUTIVE SUMMARY ==========
    if model.final_recommendation or model.risk_score:
        parts.append("\n---\n")
        parts.append("## 📋 Executive Summary")
        parts.append("")

        if model.final_recommendation:
            parts.append(f"**최종 권고:** {model.final_recommendation}")
        if model.confidence:
            parts.append(f"**신뢰도:** {model.confidence * 100:.1f}%")
        if model.risk_score:
            parts.append(f"**리스크 점수:** {model.risk_score:.2f}/100")

    # ========== SECTIONS (by priority) ==========
    # Sort sections by priority
    sorted_sections = sorted(
        model.sections.items(),
        key=lambda x: SECTION_SCHEMA.get(x[0], {}).get('priority', 999)
    )

    for section_key, section_data in sorted_sections:
        if section_data:  # Only render non-empty sections
            parts.append("\n---\n")
            parts.append(render_section(section_key, section_data))

    # ========== EXTRA FIELDS ==========
    if model.extra_fields:
        parts.append("\n---\n")
        parts.append("## 🗂️ Additional Fields")
        parts.append("")
        parts.append("*Fields not in standard schema:*")
        parts.append("")

        for key, value in sorted(model.extra_fields.items())[:30]:
            if isinstance(value, (dict, list)):
                parts.append(f"\n### {key}")
                parts.extend(render_value(value, key))
            else:
                parts.append(f"- **{key}:** {format_value(value, key)}")

        if len(model.extra_fields) > 30:
            parts.append(f"\n*... ({len(model.extra_fields) - 30}개 필드 생략)*")

    # ========== FOOTER ==========
    parts.append("\n---\n")
    parts.append("## ⚠️ Disclaimer")
    parts.append("")
    parts.append("본 리포트는 EIMAS 시스템에 의해 자동 생성되었으며, 투자 권유가 아닙니다.")
    parts.append("모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.")
    parts.append("")
    parts.append("---")
    parts.append("*Generated by EIMAS Schema-Driven Renderer v2.0*")

    return "\n".join(parts)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(raw_json: Dict[str, Any], markdown: str) -> List[str]:
    """
    Validate that markdown output matches input

    Returns list of validation errors (empty = success)
    """
    errors = []

    # Check 1: Final recommendation matches
    final_rec = raw_json.get('final_recommendation')
    if final_rec and final_rec not in markdown:
        errors.append(f"Final recommendation '{final_rec}' not found in markdown")

    # Check 2: Decision policy stance matches final recommendation
    op_report = raw_json.get('operational_report', {})
    decision_policy = op_report.get('decision_policy', {})
    policy_stance = decision_policy.get('final_stance')

    if policy_stance and final_rec and policy_stance != final_rec:
        errors.append(f"Stance mismatch: decision_policy={policy_stance}, final_recommendation={final_rec}")

    # Check 3: Risk score present
    risk_score = raw_json.get('risk_score')
    if risk_score is not None and str(risk_score) not in markdown:
        errors.append(f"Risk score {risk_score} not found in markdown")

    return errors


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def convert_json_to_md(json_path: Path) -> Path:
    """Convert JSON file to Markdown using schema-driven renderer"""

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    # Normalize
    model = normalize(raw_json)

    # Render
    markdown = render_md(model)

    # Validate
    errors = validate_output(raw_json, markdown)
    if errors:
        print("⚠️  Validation warnings:")
        for err in errors:
            print(f"  - {err}")

    # Save
    md_path = json_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    # Stats
    json_size = len(json.dumps(raw_json))
    md_size = len(markdown)
    coverage = (md_size / json_size) * 100 if json_size > 0 else 0

    print(f"✓ Converted: {json_path.name} → {md_path.name}")
    print(f"  JSON: {json_size:,} bytes, MD: {md_size:,} bytes")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Sections: {len(model.sections)}")
    print(f"  Extra fields: {len(model.extra_fields)}")

    return md_path


def main():
    """Main entry point"""
    output_dir = Path(__file__).parent.parent / "outputs"

    if len(sys.argv) > 1:
        # Specific file
        json_path = output_dir / sys.argv[1]
    else:
        # Latest eimas_*.json
        json_files = sorted(output_dir.glob("eimas_*.json"), reverse=True)
        if not json_files:
            print("No eimas_*.json files found in outputs/")
            return
        json_path = json_files[0]

    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    convert_json_to_md(json_path)


if __name__ == "__main__":
    main()
