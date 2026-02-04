#!/usr/bin/env python3
"""
Direct JSON to HTML Converter
==============================
Converts EIMAS JSON output directly to HTML without going through MD.

Advantages over MD→HTML:
- No information loss
- Better table rendering
- More control over styling
- Faster conversion
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union

# Reuse schema from MD converter
from json_to_md_converter import SECTION_SCHEMA, TABLE_COLUMN_ORDER, format_value


# ============================================================================
# HTML TEMPLATES
# ============================================================================

HTML_HEADER = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIMAS Analysis Report</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
            --border: #30363d;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }

        /* Header */
        .header {
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        .header .meta {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Executive Summary */
        .exec-summary {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
        }
        .exec-summary h2 {
            color: var(--accent-blue);
            margin-bottom: 15px;
        }
        .exec-summary .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .metric-card {
            background: var(--bg-primary);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .metric-value.bullish { color: var(--accent-green); }
        .metric-value.bearish { color: var(--accent-red); }
        .metric-value.neutral { color: var(--accent-yellow); }

        /* Sections */
        .section {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .section h2 {
            color: var(--accent-blue);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.4rem;
        }
        .section h3 {
            color: var(--text-primary);
            margin: 20px 0 10px 0;
            font-size: 1.1rem;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }
        thead {
            background: var(--bg-primary);
        }
        th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: var(--accent-blue);
            border-bottom: 2px solid var(--border);
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }
        tr:hover {
            background: rgba(56, 139, 253, 0.1);
        }

        /* Lists */
        ul { list-style: none; margin: 10px 0; }
        li {
            padding: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        li:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: var(--accent-blue);
        }

        /* Nested content */
        .nested { margin-left: 20px; padding-left: 15px; border-left: 2px solid var(--border); }

        /* Key-value pairs */
        .kv-pair {
            display: flex;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }
        .kv-key {
            flex: 0 0 200px;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .kv-value {
            flex: 1;
            color: var(--text-primary);
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-success { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .badge-danger { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
        .badge-warning { background: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }
        .badge-info { background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); }

        /* Footer */
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Empty state */
        .empty { color: var(--text-secondary); font-style: italic; }
    </style>
</head>
<body>
<div class="container">
"""

HTML_FOOTER = """
<div class="footer">
    <p>⚠️ 본 리포트는 EIMAS 시스템에 의해 자동 생성되었으며, 투자 권유가 아닙니다.</p>
    <p>모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.</p>
    <p style="margin-top: 10px; font-size: 0.8rem;">Generated by EIMAS Direct JSON→HTML Converter v2.0</p>
</div>
</div>
</body>
</html>
"""


# ============================================================================
# HTML RENDERING
# ============================================================================

def escape_html(text: str) -> str:
    """Escape HTML special characters"""
    if not isinstance(text, str):
        text = str(text)
    return (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;'))


def render_badge(text: str) -> str:
    """Render text with appropriate badge styling"""
    text_upper = text.upper()
    if 'BULLISH' in text_upper or 'BUY' in text_upper or 'ENTRY' in text_upper:
        return f'<span class="badge badge-success">{escape_html(text)}</span>'
    elif 'BEARISH' in text_upper or 'SELL' in text_upper:
        return f'<span class="badge badge-danger">{escape_html(text)}</span>'
    elif 'NEUTRAL' in text_upper or 'HOLD' in text_upper:
        return f'<span class="badge badge-warning">{escape_html(text)}</span>'
    elif 'PASSED' in text_upper or 'OK' in text_upper or 'COMPLETE' in text_upper:
        return f'<span class="badge badge-success">{escape_html(text)}</span>'
    else:
        return escape_html(text)


def render_table(data: List[Dict[str, Any]], max_rows: int = 50) -> str:
    """Render list of dicts as HTML table"""
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return '<p class="empty">(empty)</p>'

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
    ordered_keys.extend(sorted(all_keys))

    # Build table
    lines = ['<table>']

    # Header
    lines.append('<thead><tr>')
    for key in ordered_keys:
        lines.append(f'<th>{escape_html(key)}</th>')
    lines.append('</tr></thead>')

    # Body
    lines.append('<tbody>')
    for item in data[:max_rows]:
        if isinstance(item, dict):
            lines.append('<tr>')
            for key in ordered_keys:
                value = format_value(item.get(key), key)
                lines.append(f'<td>{render_badge(value)}</td>')
            lines.append('</tr>')
    lines.append('</tbody>')

    if len(data) > max_rows:
        lines.append(f'<tfoot><tr><td colspan="{len(ordered_keys)}">'
                    f'... ({len(data) - max_rows}개 생략)</td></tr></tfoot>')

    lines.append('</table>')
    return '\n'.join(lines)


def render_dict(data: Dict[str, Any], indent: int = 0) -> str:
    """Render dict as key-value pairs"""
    if not data:
        return '<p class="empty">(empty)</p>'

    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f'<div class="kv-pair"><div class="kv-key">{escape_html(key)}</div>'
                        f'<div class="kv-value">')
            lines.append(render_dict(value, indent + 1))
            lines.append('</div></div>')
        elif isinstance(value, list):
            lines.append(f'<div class="kv-pair"><div class="kv-key">{escape_html(key)}</div>'
                        f'<div class="kv-value">')
            lines.append(render_list(value, indent + 1))
            lines.append('</div></div>')
        else:
            formatted = format_value(value, key)
            lines.append(f'<div class="kv-pair"><div class="kv-key">{escape_html(key)}</div>'
                        f'<div class="kv-value">{render_badge(formatted)}</div></div>')

    return '\n'.join(lines)


def render_list(data: List[Any], indent: int = 0) -> str:
    """Render list as HTML"""
    if not data:
        return '<p class="empty">(empty)</p>'

    # Check if list of dicts -> table
    if all(isinstance(item, dict) for item in data):
        return render_table(data)

    # Otherwise bullet list
    lines = ['<ul>']
    for item in data[:30]:
        if isinstance(item, dict):
            lines.append('<li>')
            lines.append(render_dict(item, indent + 1))
            lines.append('</li>')
        elif isinstance(item, list):
            lines.append('<li>')
            lines.append(render_list(item, indent + 1))
            lines.append('</li>')
        else:
            lines.append(f'<li>{escape_html(str(item))}</li>')

    if len(data) > 30:
        lines.append(f'<li class="empty">... ({len(data) - 30}개 생략)</li>')
    lines.append('</ul>')

    return '\n'.join(lines)


def render_value(value: Any, key: str = "") -> str:
    """Render any value to HTML"""
    if value is None or value == "":
        return '<p class="empty">N/A</p>'
    elif isinstance(value, dict):
        return render_dict(value)
    elif isinstance(value, list):
        return render_list(value)
    else:
        formatted = format_value(value, key)
        return f'<p>{render_badge(formatted)}</p>'


def render_section(section_key: str, section_data: Any) -> str:
    """Render a section to HTML"""
    schema = SECTION_SCHEMA.get(section_key, {})
    title = schema.get('title', section_key)

    html = f'<div class="section">\n<h2>{title}</h2>\n'
    html += render_value(section_data, section_key)
    html += '\n</div>'

    return html


# ============================================================================
# MAIN CONVERTER
# ============================================================================

def convert_json_to_html(json_path: Path) -> Path:
    """Convert JSON file to HTML"""

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Start HTML
    html_parts = [HTML_HEADER]

    # ========== HEADER ==========
    ts = data.get('timestamp', '')[:19].replace('T', ' ')
    schema_version = data.get('schema_version', '1.0.0')

    html_parts.append(f'''
<div class="header">
    <h1>📊 EIMAS 분석 리포트</h1>
    <div class="meta">
        <span>생성 시간: {escape_html(ts)}</span> •
        <span>스키마 버전: {escape_html(schema_version)}</span>
    </div>
</div>
''')

    # ========== EXECUTIVE SUMMARY ==========
    final_rec = data.get('final_recommendation')
    confidence = data.get('confidence')
    risk_score = data.get('risk_score')

    if final_rec or risk_score:
        stance_class = 'neutral'
        if final_rec:
            if 'BULLISH' in final_rec.upper():
                stance_class = 'bullish'
            elif 'BEARISH' in final_rec.upper():
                stance_class = 'bearish'

        html_parts.append(f'''
<div class="exec-summary">
    <h2>📋 Executive Summary</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">최종 권고</div>
            <div class="metric-value {stance_class}">{escape_html(final_rec or 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">신뢰도</div>
            <div class="metric-value">{(confidence or 0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">리스크 점수</div>
            <div class="metric-value">{risk_score or 0:.2f}/100</div>
        </div>
    </div>
</div>
''')

    # ========== SECTIONS ==========
    # Extract sections from operational_report
    sections = {}
    for key in SECTION_SCHEMA.keys():
        if key in data:
            sections[key] = data[key]

    # Check operational_report
    if 'operational_report' in data and isinstance(data['operational_report'], dict):
        for key in SECTION_SCHEMA.keys():
            if key in data['operational_report']:
                sections[key] = data['operational_report'][key]

    # Sort and render
    sorted_sections = sorted(
        sections.items(),
        key=lambda x: SECTION_SCHEMA.get(x[0], {}).get('priority', 999)
    )

    for section_key, section_data in sorted_sections:
        if section_data:
            html_parts.append(render_section(section_key, section_data))

    # ========== FOOTER ==========
    html_parts.append(HTML_FOOTER)

    # Save
    html_path = json_path.with_suffix('.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    # Stats
    json_size = len(json.dumps(data))
    html_size = len('\n'.join(html_parts))

    print(f"✓ Converted: {json_path.name} → {html_path.name}")
    print(f"  JSON: {json_size:,} bytes, HTML: {html_size:,} bytes")
    print(f"  Sections: {len(sections)}")

    return html_path


def main():
    """Main entry point"""
    output_dir = Path(__file__).parent.parent / "outputs"

    if len(sys.argv) > 1:
        json_path = output_dir / sys.argv[1]
    else:
        json_files = sorted(output_dir.glob("eimas_*.json"), reverse=True)
        if not json_files:
            print("No eimas_*.json files found in outputs/")
            return
        json_path = json_files[0]

    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    convert_json_to_html(json_path)


if __name__ == "__main__":
    main()
