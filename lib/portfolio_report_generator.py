#!/usr/bin/env python3
"""
Portfolio Report Generator
==========================
EIMAS 분석 결과를 포트폴리오 매니저/인사팀이 볼 수 있는 HTML 리포트로 생성.

주요 섹션:
1. Executive Summary - 최종 권고, 신뢰도, 리스크
2. Market Regime - 시장 레짐 분석
3. Multi-Agent Debate - AI 에이전트 토론 과정
4. Risk Analysis - 자산군별 위험
5. Detailed Signals - 상세 시그널 (action_guide + theory_note)
6. Asset Risk Metrics - 자산별 리스크 메트릭 (Sharpe/Sortino/VaR/MDD)
7. Events & News - 뉴스 및 이벤트
8. Scenario Analysis - 시나리오별 전망
9. Sector Recommendations - 섹터 권고
10. Portfolio Strategy - 포트폴리오 전략
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd

# 새 모듈 import (signal_analyzer, asset_risk_metrics)
try:
    from lib.signal_analyzer import SignalAnalyzer, Signal
    from lib.asset_risk_metrics import AssetRiskCalculator, AssetRiskMetrics
except ImportError:
    try:
        from signal_analyzer import SignalAnalyzer, Signal
        from asset_risk_metrics import AssetRiskCalculator, AssetRiskMetrics
    except ImportError:
        SignalAnalyzer = None
        AssetRiskCalculator = None


# ============================================================================
# 색상 테마
# ============================================================================

COLORS = {
    'bg_primary': '#0d1117',
    'bg_secondary': '#161b22',
    'bg_card': '#21262d',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',
    'accent_blue': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_yellow': '#d29922',
    'accent_purple': '#a371f7',
    'border': '#30363d',
    'gradient_bull': 'linear-gradient(135deg, #238636 0%, #2ea043 100%)',
    'gradient_bear': 'linear-gradient(135deg, #da3633 0%, #f85149 100%)',
    'gradient_neutral': 'linear-gradient(135deg, #9e6a03 0%, #d29922 100%)',
}


# ============================================================================
# 메인 리포트 생성 함수
# ============================================================================

def generate_portfolio_report(
    integrated_data: Dict,
    ai_report_data: Dict = None,
    output_path: str = None,
    signals_data: List[Dict] = None,
    metrics_data: Dict = None,
) -> str:
    """
    통합 포트폴리오 리포트 HTML 생성

    Args:
        integrated_data: integrated_*.json 데이터
        ai_report_data: ai_report_*.json 데이터 (선택)
        output_path: 출력 파일 경로 (선택)
        signals_data: 상세 시그널 데이터 (선택, 외부에서 직접 전달)
        metrics_data: 자산별 리스크 메트릭 (선택, 외부에서 직접 전달)

    Returns:
        HTML 문자열
    """
    ai_report_data = ai_report_data or {}
    timestamp = integrated_data.get('timestamp', datetime.now().isoformat())

    html_parts = [
        _generate_html_head(timestamp),
        _generate_header(integrated_data, ai_report_data),
        _generate_executive_summary(integrated_data, ai_report_data),
        _generate_market_regime_section(integrated_data),
        _generate_multi_agent_debate_section(integrated_data, ai_report_data),
        _generate_risk_analysis_section(integrated_data),
        _generate_detailed_signals_section(integrated_data, signals_data),  # NEW
        _generate_asset_risk_metrics_section(integrated_data, metrics_data),  # NEW
        _generate_events_news_section(integrated_data, ai_report_data),
        _generate_scenario_section(ai_report_data),
        _generate_sector_recommendations_section(ai_report_data),
        _generate_portfolio_strategy_section(integrated_data, ai_report_data),
        _generate_technical_indicators_section(ai_report_data),
        _generate_ai_analysis_section(ai_report_data),
        _generate_footer(),
        '</div></body></html>'
    ]

    html = '\n'.join(html_parts)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    return html


# ============================================================================
# HTML 헤더
# ============================================================================

def _generate_html_head(timestamp: str) -> str:
    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIMAS Portfolio Report - {timestamp[:10]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}

        /* Header */
        .report-header {{
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            text-align: center;
        }}

        .report-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['accent_blue']};
            margin-bottom: 8px;
        }}

        .report-subtitle {{
            color: {COLORS['text_secondary']};
            font-size: 1rem;
        }}

        /* Executive Summary Banner */
        .executive-banner {{
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
        }}

        .executive-banner.bullish {{
            background: {COLORS['gradient_bull']};
        }}

        .executive-banner.bearish {{
            background: {COLORS['gradient_bear']};
        }}

        .executive-banner.neutral {{
            background: {COLORS['gradient_neutral']};
        }}

        .summary-item {{
            text-align: center;
        }}

        .summary-label {{
            font-size: 0.85rem;
            opacity: 0.9;
            margin-bottom: 4px;
        }}

        .summary-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}

        /* Section */
        .section {{
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 12px;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .section-icon {{
            font-size: 1.5rem;
        }}

        /* Cards Grid */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}

        .card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .card-title {{
            font-weight: 600;
            color: {COLORS['text_primary']};
        }}

        .card-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-bullish {{ background: rgba(63, 185, 80, 0.2); color: {COLORS['accent_green']}; }}
        .badge-bearish {{ background: rgba(248, 81, 73, 0.2); color: {COLORS['accent_red']}; }}
        .badge-neutral {{ background: rgba(210, 153, 34, 0.2); color: {COLORS['accent_yellow']}; }}
        .badge-info {{ background: rgba(88, 166, 255, 0.2); color: {COLORS['accent_blue']}; }}

        /* Agent Cards */
        .agent-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}

        .agent-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }}

        .agent-card.bullish::before {{ background: {COLORS['accent_green']}; }}
        .agent-card.bearish::before {{ background: {COLORS['accent_red']}; }}
        .agent-card.neutral::before {{ background: {COLORS['accent_yellow']}; }}

        .agent-name {{
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 8px;
        }}

        .agent-position {{
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .agent-confidence {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
        }}

        .agent-reasoning {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid {COLORS['border']};
            color: {COLORS['text_secondary']};
            font-size: 0.85rem;
            line-height: 1.5;
        }}

        /* Debate Flow */
        .debate-flow {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-top: 20px;
        }}

        .debate-round {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 16px;
        }}

        .debate-round-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            color: {COLORS['accent_blue']};
            font-weight: 600;
        }}

        .debate-message {{
            padding: 12px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 6px;
            margin-bottom: 8px;
        }}

        .debate-agent {{
            font-weight: 600;
            color: {COLORS['accent_purple']};
            margin-bottom: 4px;
        }}

        /* Risk Meter */
        .risk-meter {{
            background: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}

        .risk-bar {{
            height: 12px;
            background: {COLORS['bg_primary']};
            border-radius: 6px;
            overflow: hidden;
            margin: 12px 0;
        }}

        .risk-fill {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
        }}

        .risk-labels {{
            display: flex;
            justify-content: space-between;
            color: {COLORS['text_muted']};
            font-size: 0.75rem;
        }}

        /* News Card */
        .news-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 12px;
        }}

        .news-content {{
            color: {COLORS['text_secondary']};
            line-height: 1.7;
            white-space: pre-wrap;
        }}

        .news-content h3 {{
            color: {COLORS['text_primary']};
            margin: 16px 0 8px 0;
            font-size: 1rem;
        }}

        /* Scenario Cards */
        .scenario-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
        }}

        .scenario-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .scenario-name {{
            font-weight: 600;
            font-size: 1.1rem;
        }}

        .scenario-prob {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .scenario-target {{
            color: {COLORS['text_secondary']};
            margin-bottom: 12px;
        }}

        .scenario-triggers {{
            list-style: none;
            padding: 0;
        }}

        .scenario-triggers li {{
            padding: 4px 0;
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
        }}

        .scenario-triggers li::before {{
            content: '→ ';
            color: {COLORS['accent_blue']};
        }}

        /* Sector Table */
        .sector-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .sector-table th,
        .sector-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .sector-table th {{
            color: {COLORS['text_secondary']};
            font-weight: 500;
            font-size: 0.85rem;
        }}

        .rating-overweight {{ color: {COLORS['accent_green']}; font-weight: 600; }}
        .rating-neutral {{ color: {COLORS['accent_yellow']}; font-weight: 600; }}
        .rating-underweight {{ color: {COLORS['accent_red']}; font-weight: 600; }}

        /* Entry Exit Strategy */
        .strategy-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .strategy-box {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
        }}

        .strategy-box h4 {{
            color: {COLORS['accent_blue']};
            margin-bottom: 16px;
            font-size: 1rem;
        }}

        .price-level {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .price-level:last-child {{
            border-bottom: none;
        }}

        /* Technical Indicators */
        .indicator-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }}

        .indicator-item {{
            background: {COLORS['bg_card']};
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}

        .indicator-label {{
            color: {COLORS['text_muted']};
            font-size: 0.8rem;
            margin-bottom: 4px;
        }}

        .indicator-value {{
            font-size: 1.2rem;
            font-weight: 600;
        }}

        /* AI Analysis */
        .ai-analysis {{
            background: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 24px;
            line-height: 1.8;
        }}

        .ai-analysis h1, .ai-analysis h2, .ai-analysis h3 {{
            color: {COLORS['accent_blue']};
            margin: 20px 0 12px 0;
        }}

        .ai-analysis h1 {{ font-size: 1.5rem; }}
        .ai-analysis h2 {{ font-size: 1.25rem; }}
        .ai-analysis h3 {{ font-size: 1.1rem; }}

        .ai-analysis ul, .ai-analysis ol {{
            margin-left: 20px;
            margin-bottom: 12px;
        }}

        .ai-analysis li {{
            margin-bottom: 6px;
        }}

        .ai-analysis strong {{
            color: {COLORS['accent_green']};
        }}

        /* Footer */
        .report-footer {{
            text-align: center;
            padding: 32px;
            color: {COLORS['text_muted']};
            font-size: 0.85rem;
            border-top: 1px solid {COLORS['border']};
            margin-top: 40px;
        }}

        .data-sources {{
            margin-top: 16px;
            color: {COLORS['text_muted']};
            font-size: 0.8rem;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .container {{ padding: 16px; }}
            .report-title {{ font-size: 1.8rem; }}
            .executive-banner {{ padding: 20px; }}
            .summary-value {{ font-size: 1.4rem; }}
        }}

        /* Charts */
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}

        /* Consensus Box */
        .consensus-box {{
            background: linear-gradient(135deg, {COLORS['bg_card']} 0%, rgba(88, 166, 255, 0.1) 100%);
            border: 2px solid {COLORS['accent_blue']};
            border-radius: 12px;
            padding: 24px;
            margin-top: 24px;
            text-align: center;
        }}

        .consensus-title {{
            color: {COLORS['accent_blue']};
            font-size: 1.1rem;
            margin-bottom: 8px;
        }}

        .consensus-result {{
            font-size: 2rem;
            font-weight: 700;
        }}

        /* Conflict Warning */
        .conflict-warning {{
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid {COLORS['accent_red']};
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
        }}

        .conflict-title {{
            color: {COLORS['accent_red']};
            font-weight: 600;
            margin-bottom: 8px;
        }}

        /* Signal Cards */
        .signal-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            position: relative;
            overflow: hidden;
        }}

        .signal-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }}

        .signal-card.critical::before {{ background: {COLORS['accent_red']}; }}
        .signal-card.alert::before {{ background: {COLORS['accent_yellow']}; }}
        .signal-card.warning::before {{ background: {COLORS['accent_blue']}; }}

        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .signal-ticker {{
            font-weight: 700;
            font-size: 1.1rem;
            color: {COLORS['accent_blue']};
        }}

        .signal-level {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .level-critical {{ background: rgba(248, 81, 73, 0.2); color: {COLORS['accent_red']}; }}
        .level-alert {{ background: rgba(210, 153, 34, 0.2); color: {COLORS['accent_yellow']}; }}
        .level-warning {{ background: rgba(88, 166, 255, 0.2); color: {COLORS['accent_blue']}; }}

        .signal-indicator {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}

        .signal-value {{
            display: flex;
            gap: 16px;
            margin-bottom: 12px;
        }}

        .signal-value-item {{
            background: rgba(255, 255, 255, 0.03);
            padding: 8px 12px;
            border-radius: 6px;
        }}

        .signal-value-label {{
            color: {COLORS['text_muted']};
            font-size: 0.75rem;
        }}

        .signal-value-number {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .signal-guide {{
            background: rgba(63, 185, 80, 0.1);
            border-left: 3px solid {COLORS['accent_green']};
            padding: 12px;
            border-radius: 0 6px 6px 0;
            margin-bottom: 12px;
        }}

        .signal-guide-title {{
            color: {COLORS['accent_green']};
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }}

        .signal-guide-text {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        .signal-theory {{
            background: rgba(163, 113, 247, 0.1);
            border-left: 3px solid {COLORS['accent_purple']};
            padding: 12px;
            border-radius: 0 6px 6px 0;
        }}

        .signal-theory-title {{
            color: {COLORS['accent_purple']};
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }}

        .signal-theory-text {{
            color: {COLORS['text_secondary']};
            font-size: 0.85rem;
            line-height: 1.5;
            font-style: italic;
        }}

        /* Risk Metrics Table */
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 0.9rem;
        }}

        .metrics-table th {{
            background: {COLORS['bg_primary']};
            color: {COLORS['text_secondary']};
            font-weight: 500;
            padding: 12px 8px;
            text-align: left;
            border-bottom: 2px solid {COLORS['border']};
            position: sticky;
            top: 0;
        }}

        .metrics-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .metrics-table tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .metric-positive {{ color: {COLORS['accent_green']}; }}
        .metric-negative {{ color: {COLORS['accent_red']}; }}
        .metric-neutral {{ color: {COLORS['text_primary']}; }}

        .metric-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .metric-bar-fill {{
            height: 6px;
            border-radius: 3px;
            background: {COLORS['accent_blue']};
        }}

        .metrics-legend {{
            display: flex;
            gap: 24px;
            margin-top: 16px;
            padding: 12px;
            background: {COLORS['bg_card']};
            border-radius: 8px;
            font-size: 0.85rem;
        }}

        .metrics-legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        /* Signal Summary */
        .signal-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}

        .signal-summary-card {{
            background: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}

        .signal-count {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .signal-count-label {{
            color: {COLORS['text_muted']};
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
<div class="container">
'''


# ============================================================================
# Header
# ============================================================================

def _generate_header(integrated_data: Dict, ai_report_data: Dict) -> str:
    timestamp = integrated_data.get('timestamp', '')
    return f'''
    <div class="report-header">
        <h1 class="report-title">EIMAS Portfolio Report</h1>
        <p class="report-subtitle">Economic Intelligence Multi-Agent System | Generated: {timestamp}</p>
    </div>
    '''


# ============================================================================
# Executive Summary
# ============================================================================

def _generate_executive_summary(integrated_data: Dict, ai_report_data: Dict) -> str:
    recommendation = integrated_data.get('final_recommendation', 'NEUTRAL')
    confidence = integrated_data.get('confidence', 0)
    risk_score = integrated_data.get('risk_score', 0)
    risk_level = integrated_data.get('risk_level', 'MEDIUM')

    # 권고에 따른 클래스 결정
    rec_lower = recommendation.lower()
    if 'bull' in rec_lower or 'buy' in rec_lower:
        banner_class = 'bullish'
    elif 'bear' in rec_lower or 'sell' in rec_lower:
        banner_class = 'bearish'
    else:
        banner_class = 'neutral'

    return f'''
    <div class="executive-banner {banner_class}">
        <div class="summary-item">
            <div class="summary-label">Final Recommendation</div>
            <div class="summary-value">{recommendation}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Confidence</div>
            <div class="summary-value">{confidence:.1%}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Risk Score</div>
            <div class="summary-value">{risk_score:.1f}/100</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Risk Level</div>
            <div class="summary-value">{risk_level}</div>
        </div>
    </div>
    '''


# ============================================================================
# Market Regime Section
# ============================================================================

def _generate_market_regime_section(integrated_data: Dict) -> str:
    regime_data = integrated_data.get('regime', {})
    fred_summary = integrated_data.get('fred_summary', {})

    regime = regime_data.get('regime', 'Unknown')
    trend = regime_data.get('trend', 'Unknown')
    volatility = regime_data.get('volatility', 'Unknown')
    regime_confidence = regime_data.get('confidence', 0)
    description = regime_data.get('description', '')
    strategy = regime_data.get('strategy', '')

    # FRED 데이터
    rrp = fred_summary.get('rrp', 0)
    tga = fred_summary.get('tga', 0)
    net_liquidity = fred_summary.get('net_liquidity', 0)
    fed_funds = fred_summary.get('fed_funds', 0)
    treasury_10y = fred_summary.get('treasury_10y', 0)
    spread = fred_summary.get('spread_10y2y', 0)

    # 레짐 색상
    if 'bull' in regime.lower():
        regime_color = COLORS['accent_green']
    elif 'bear' in regime.lower():
        regime_color = COLORS['accent_red']
    else:
        regime_color = COLORS['accent_yellow']

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">📈</span>
            Market Regime Analysis
        </h2>

        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Current Regime</span>
                </div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {regime_color}; margin-bottom: 8px;">
                    {regime}
                </div>
                <div style="color: {COLORS['text_secondary']}; margin-bottom: 4px;">
                    Trend: {trend}
                </div>
                <div style="color: {COLORS['text_secondary']}; margin-bottom: 4px;">
                    Volatility: {volatility}
                </div>
                <div style="color: {COLORS['text_secondary']};">
                    Confidence: {regime_confidence:.0%}
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Liquidity Status</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="color: {COLORS['text_secondary']}; font-size: 0.85rem;">Net Liquidity</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">${net_liquidity:,.1f}B</div>
                </div>
                <div style="display: flex; gap: 20px;">
                    <div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">RRP</div>
                        <div>${rrp:,.1f}B</div>
                    </div>
                    <div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">TGA</div>
                        <div>${tga:,.1f}B</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Interest Rates</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">Fed Funds</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{fed_funds:.2f}%</div>
                    </div>
                    <div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">10Y Treasury</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{treasury_10y:.2f}%</div>
                    </div>
                    <div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">10Y-2Y Spread</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{spread:.2f}%</div>
                    </div>
                </div>
            </div>
        </div>

        {f'<div style="margin-top: 16px; padding: 16px; background: {COLORS["bg_card"]}; border-radius: 8px;"><strong>전략:</strong> {strategy}</div>' if strategy else ''}
    </div>
    '''


# ============================================================================
# Multi-Agent Debate Section
# ============================================================================

def _generate_multi_agent_debate_section(integrated_data: Dict, ai_report_data: Dict) -> str:
    full_mode = integrated_data.get('full_mode_position', 'N/A')
    ref_mode = integrated_data.get('reference_mode_position', 'N/A')
    modes_agree = integrated_data.get('modes_agree', False)
    dissent_records = integrated_data.get('dissent_records', [])
    has_strong_dissent = integrated_data.get('has_strong_dissent', False)

    # 에이전트 카드 클래스 결정
    def get_position_class(pos):
        pos_lower = pos.lower() if pos else ''
        if 'bull' in pos_lower or 'buy' in pos_lower:
            return 'bullish'
        elif 'bear' in pos_lower or 'sell' in pos_lower:
            return 'bearish'
        return 'neutral'

    full_class = get_position_class(full_mode)
    ref_class = get_position_class(ref_mode)

    # 합의 상태
    if modes_agree:
        consensus_html = f'''
        <div class="consensus-box">
            <div class="consensus-title">✅ Agent Consensus Reached</div>
            <div class="consensus-result" style="color: {COLORS['accent_green']};">{full_mode}</div>
            <div style="color: {COLORS['text_secondary']}; margin-top: 8px;">
                Both analysis modes (FULL 365-day & REFERENCE 90-day) agree on the market outlook.
            </div>
        </div>
        '''
    else:
        consensus_html = f'''
        <div class="consensus-box" style="border-color: {COLORS['accent_yellow']};">
            <div class="consensus-title" style="color: {COLORS['accent_yellow']};">⚠️ Divergent Views</div>
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 12px;">
                <div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">FULL Mode (365일)</div>
                    <div style="font-size: 1.3rem; font-weight: 600;">{full_mode}</div>
                </div>
                <div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">REF Mode (90일)</div>
                    <div style="font-size: 1.3rem; font-weight: 600;">{ref_mode}</div>
                </div>
            </div>
        </div>
        '''

    # Dissent 레코드
    dissent_html = ''
    if dissent_records:
        dissent_items = ''.join([
            f'<div class="debate-message"><div class="debate-agent">{d.get("agent", "Agent")}</div>{d.get("reason", "No reason provided")}</div>'
            for d in dissent_records[:5]
        ])
        dissent_html = f'''
        <div class="conflict-warning">
            <div class="conflict-title">⚠️ Dissent Records</div>
            {dissent_items}
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">🤖</span>
            Multi-Agent Debate Analysis
        </h2>

        <p style="color: {COLORS['text_secondary']}; margin-bottom: 20px;">
            AI 에이전트들이 서로 다른 데이터 기간을 분석하여 독립적인 의견을 제시하고, 토론을 통해 합의에 도달합니다.
        </p>

        <div class="cards-grid">
            <div class="agent-card {full_class}">
                <div class="agent-name">📊 FULL Mode Agent</div>
                <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 8px;">365일 장기 데이터 분석</div>
                <div class="agent-position" style="color: {COLORS['accent_green'] if full_class == 'bullish' else COLORS['accent_red'] if full_class == 'bearish' else COLORS['accent_yellow']};">
                    {full_mode}
                </div>
                <div class="agent-reasoning">
                    장기 추세, 구조적 변화, 거시경제 사이클을 고려한 분석 결과입니다.
                </div>
            </div>

            <div class="agent-card {ref_class}">
                <div class="agent-name">⚡ REFERENCE Mode Agent</div>
                <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 8px;">90일 단기 데이터 분석</div>
                <div class="agent-position" style="color: {COLORS['accent_green'] if ref_class == 'bullish' else COLORS['accent_red'] if ref_class == 'bearish' else COLORS['accent_yellow']};">
                    {ref_mode}
                </div>
                <div class="agent-reasoning">
                    최근 모멘텀, 단기 이벤트, 시장 심리를 반영한 분석 결과입니다.
                </div>
            </div>
        </div>

        {consensus_html}
        {dissent_html}
    </div>
    '''


# ============================================================================
# Risk Analysis Section
# ============================================================================

def _generate_risk_analysis_section(integrated_data: Dict) -> str:
    risk_score = integrated_data.get('risk_score', 0)
    base_risk = integrated_data.get('base_risk_score', risk_score)
    micro_adj = integrated_data.get('microstructure_adjustment', 0)
    bubble_adj = integrated_data.get('bubble_risk_adjustment', 0)
    market_quality = integrated_data.get('market_quality') or {}
    bubble_risk = integrated_data.get('bubble_risk') or {}

    # 리스크 색상
    if risk_score < 30:
        risk_color = COLORS['accent_green']
        risk_label = 'LOW RISK'
    elif risk_score < 60:
        risk_color = COLORS['accent_yellow']
        risk_label = 'MEDIUM RISK'
    else:
        risk_color = COLORS['accent_red']
        risk_label = 'HIGH RISK'

    # Market Quality
    avg_liquidity = market_quality.get('avg_liquidity_score', 'N/A') if market_quality else 'N/A'
    high_toxicity = market_quality.get('high_toxicity_tickers', []) if market_quality else []

    # Bubble Risk
    bubble_status = bubble_risk.get('overall_status', 'N/A') if bubble_risk else 'N/A'
    risk_tickers = bubble_risk.get('risk_tickers', []) if bubble_risk else []

    # Bubble 색상
    bubble_colors = {
        'NONE': COLORS['accent_green'],
        'WATCH': COLORS['accent_yellow'],
        'WARNING': COLORS['accent_red'],
        'DANGER': '#dc2626'
    }
    bubble_color = bubble_colors.get(bubble_status, COLORS['text_muted'])

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">⚠️</span>
            Risk Analysis
        </h2>

        <div class="risk-meter">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1rem; font-weight: 600;">Overall Risk Score</span>
                <span style="font-size: 1.8rem; font-weight: 700; color: {risk_color};">{risk_score:.1f}/100</span>
            </div>
            <div class="risk-bar">
                <div class="risk-fill" style="width: {min(risk_score, 100)}%; background: {risk_color};"></div>
            </div>
            <div class="risk-labels">
                <span>Low Risk</span>
                <span>Medium</span>
                <span>High Risk</span>
            </div>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Risk Score Breakdown</span>
                </div>
                <div style="margin-top: 12px;">
                    <div class="price-level">
                        <span>Base Risk (CriticalPath)</span>
                        <span style="font-weight: 600;">{base_risk:.1f}</span>
                    </div>
                    <div class="price-level">
                        <span>Microstructure Adjustment</span>
                        <span style="font-weight: 600; color: {COLORS['accent_green'] if micro_adj < 0 else COLORS['accent_red'] if micro_adj > 0 else COLORS['text_primary']};">
                            {micro_adj:+.1f}
                        </span>
                    </div>
                    <div class="price-level">
                        <span>Bubble Risk Adjustment</span>
                        <span style="font-weight: 600; color: {COLORS['accent_red'] if bubble_adj > 0 else COLORS['text_primary']};">
                            {bubble_adj:+.1f}
                        </span>
                    </div>
                    <div class="price-level" style="border-top: 2px solid {COLORS['border']}; margin-top: 8px; padding-top: 8px;">
                        <span style="font-weight: 600;">Final Risk Score</span>
                        <span style="font-weight: 700; color: {risk_color};">{risk_score:.1f}</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Market Quality</span>
                    <span class="card-badge badge-info">Microstructure</span>
                </div>
                <div style="margin-top: 12px;">
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">Average Liquidity Score</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">{avg_liquidity if isinstance(avg_liquidity, str) else f'{avg_liquidity:.1f}'}/100</div>
                </div>
                {f'<div style="margin-top: 12px; color: {COLORS["accent_red"]}; font-size: 0.85rem;">⚠️ High Toxicity: {", ".join(high_toxicity[:3])}</div>' if high_toxicity else ''}
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Bubble Risk</span>
                    <span class="card-badge" style="background: {bubble_color}20; color: {bubble_color};">{bubble_status}</span>
                </div>
                <div style="margin-top: 12px;">
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">Greenwood-Shleifer 버블 탐지</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {bubble_color}; margin-top: 4px;">{bubble_status}</div>
                </div>
                {_format_risk_tickers(risk_tickers)}
            </div>
        </div>
    </div>
    '''


def _format_risk_tickers(tickers: List) -> str:
    if not tickers:
        return ''

    items = []
    for t in tickers[:3]:
        if isinstance(t, dict):
            ticker = t.get('ticker', 'Unknown')
            score = t.get('score', 0)
            items.append(f'{ticker}: {score:.0f}')
        else:
            items.append(str(t))

    return f'<div style="margin-top: 12px; color: {COLORS["text_secondary"]}; font-size: 0.85rem;">Top Risks: {", ".join(items)}</div>'


# ============================================================================
# Events & News Section
# ============================================================================

def _generate_events_news_section(integrated_data: Dict, ai_report_data: Dict) -> str:
    events = integrated_data.get('events_detected', [])
    perplexity_news = ai_report_data.get('perplexity_news', '')
    notable_stocks = ai_report_data.get('notable_stocks', [])

    # Events HTML
    events_html = ''
    if events:
        event_items = ''.join([
            f'<div style="padding: 8px 0; border-bottom: 1px solid {COLORS["border"]};">{e.get("description", str(e))}</div>'
            for e in events[:5]
        ])
        events_html = f'''
        <div class="card">
            <div class="card-header">
                <span class="card-title">Detected Events</span>
                <span class="card-badge badge-info">{len(events)} events</span>
            </div>
            {event_items if event_items else '<div style="color: ' + COLORS['text_muted'] + ';">No significant events detected</div>'}
        </div>
        '''

    # Notable Stocks HTML
    notable_html = ''
    if notable_stocks:
        stock_items = ''
        for stock in notable_stocks[:4]:
            ticker = stock.get('ticker', '')
            reason = stock.get('notable_reason', '')
            analysis = stock.get('deep_analysis', '')[:200] + '...' if stock.get('deep_analysis', '') else ''

            stock_items += f'''
            <div class="card">
                <div class="card-header">
                    <span class="card-title">{ticker}</span>
                    <span class="card-badge badge-neutral">{reason}</span>
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem; line-height: 1.6;">
                    {analysis}
                </div>
            </div>
            '''
        notable_html = f'<div class="cards-grid">{stock_items}</div>'

    # Perplexity News HTML
    news_html = ''
    if perplexity_news:
        # Markdown을 간단히 HTML로 변환
        news_formatted = perplexity_news.replace('### ', '<h3>').replace('\n', '</h3>\n', 1)
        news_formatted = news_formatted.replace('### ', '<h3>').replace('\n\n', '</p><p>')
        news_formatted = news_formatted.replace('**', '<strong>').replace('**', '</strong>')

        news_html = f'''
        <div class="news-card">
            <div class="card-header">
                <span class="card-title">📰 Market News (Perplexity AI)</span>
            </div>
            <div class="news-content">{perplexity_news}</div>
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">📰</span>
            Events & Market News
        </h2>

        {notable_html}
        {events_html}
        {news_html if news_html else f'<div style="color: {COLORS["text_muted"]};">No news data available. Run with --report flag to fetch news.</div>'}
    </div>
    '''


# ============================================================================
# Scenario Section
# ============================================================================

def _generate_scenario_section(ai_report_data: Dict) -> str:
    scenarios = ai_report_data.get('scenarios', [])

    if not scenarios:
        return ''

    scenario_cards = ''
    for scenario in scenarios:
        name = scenario.get('name', 'Unknown')
        prob = scenario.get('probability', 0)
        expected_return = scenario.get('expected_return', 'N/A')
        target = scenario.get('sp500_target', 'N/A')
        strategy = scenario.get('strategy', '')
        triggers = scenario.get('key_triggers', [])

        # 시나리오 색상
        if 'bull' in name.lower():
            color = COLORS['accent_green']
        elif 'bear' in name.lower():
            color = COLORS['accent_red']
        else:
            color = COLORS['accent_blue']

        triggers_html = ''.join([f'<li>{t}</li>' for t in triggers[:4]])

        scenario_cards += f'''
        <div class="scenario-card">
            <div class="scenario-header">
                <span class="scenario-name" style="color: {color};">{name}</span>
                <span class="scenario-prob" style="color: {color};">{prob}%</span>
            </div>
            <div class="scenario-target">
                <div>Expected Return: <strong>{expected_return}</strong></div>
                <div>S&P 500 Target: <strong>{target}</strong></div>
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem; margin-bottom: 12px;">
                {strategy}
            </div>
            <div style="font-size: 0.85rem; color: {COLORS['text_muted']};">Key Triggers:</div>
            <ul class="scenario-triggers">{triggers_html}</ul>
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">🎯</span>
            Scenario Analysis
        </h2>
        <div class="cards-grid">{scenario_cards}</div>
    </div>
    '''


# ============================================================================
# Sector Recommendations Section
# ============================================================================

def _generate_sector_recommendations_section(ai_report_data: Dict) -> str:
    sector_recs = ai_report_data.get('sector_recommendations', {})

    if not sector_recs:
        return ''

    bullish = sector_recs.get('bullish_sectors', [])
    neutral = sector_recs.get('neutral_sectors', [])
    bearish = sector_recs.get('bearish_sectors', [])
    hot_industries = sector_recs.get('hot_industries', [])

    # 테이블 행 생성
    rows = ''
    for s in bullish:
        etfs = ', '.join(s.get('etfs', [])[:2])
        rows += f'''
        <tr>
            <td>{s.get('name', '')}</td>
            <td class="rating-overweight">Overweight ▲</td>
            <td>{s.get('rationale', '')}</td>
            <td>{etfs}</td>
        </tr>
        '''

    for s in neutral:
        rows += f'''
        <tr>
            <td>{s.get('name', '')}</td>
            <td class="rating-neutral">Neutral ●</td>
            <td>{s.get('rationale', '')}</td>
            <td>-</td>
        </tr>
        '''

    for s in bearish:
        rows += f'''
        <tr>
            <td>{s.get('name', '')}</td>
            <td class="rating-underweight">Underweight ▼</td>
            <td>{s.get('rationale', '')}</td>
            <td>-</td>
        </tr>
        '''

    # Hot Industries
    hot_html = ''
    if hot_industries:
        hot_items = ''.join([
            f'<span style="background: {COLORS["accent_purple"]}20; color: {COLORS["accent_purple"]}; padding: 4px 12px; border-radius: 20px; margin-right: 8px; font-size: 0.85rem;">{h.get("name", "")}</span>'
            for h in hot_industries[:4]
        ])
        hot_html = f'''
        <div style="margin-top: 20px;">
            <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 8px;">🔥 Hot Industries:</div>
            {hot_items}
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">🏭</span>
            Sector Recommendations
        </h2>

        <table class="sector-table">
            <thead>
                <tr>
                    <th>Sector</th>
                    <th>Rating</th>
                    <th>Rationale</th>
                    <th>ETFs</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        {hot_html}
    </div>
    '''


# ============================================================================
# Portfolio Strategy Section
# ============================================================================

def _generate_portfolio_strategy_section(integrated_data: Dict, ai_report_data: Dict) -> str:
    entry_exit = ai_report_data.get('entry_exit_strategy', {})
    adaptive = integrated_data.get('adaptive_portfolios', {})
    gpt_recs = ai_report_data.get('gpt_recommendations', '')

    if not entry_exit and not adaptive:
        return ''

    # Entry levels
    entry_html = ''
    if entry_exit.get('entry_levels'):
        entry_items = ''.join([
            f'<div class="price-level"><span>{e.get("name", "")}</span><span>${e.get("price", 0):,.2f} ({e.get("ratio", 0)}%)</span></div>'
            for e in entry_exit.get('entry_levels', [])
        ])
        entry_html = f'''
        <div class="strategy-box">
            <h4>📥 Entry Levels</h4>
            {entry_items}
        </div>
        '''

    # Take profit levels
    tp_html = ''
    if entry_exit.get('take_profit_levels'):
        tp_items = ''.join([
            f'<div class="price-level"><span>{t.get("name", "")} ({t.get("target", "")})</span><span>${t.get("price", 0):,.2f}</span></div>'
            for t in entry_exit.get('take_profit_levels', [])
        ])
        tp_html = f'''
        <div class="strategy-box">
            <h4>📤 Take Profit Levels</h4>
            {tp_items}
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid {COLORS['border']}; color: {COLORS['accent_red']};">
                Stop Loss: ${entry_exit.get('stop_loss_level', 0):,.2f} ({entry_exit.get('stop_loss_percent', 0):.1f}%)
            </div>
        </div>
        '''

    # Adaptive portfolios
    adaptive_html = ''
    if adaptive:
        adaptive_items = ''.join([
            f'<div class="price-level"><span style="text-transform: capitalize;">{k}</span><span style="color: {COLORS["accent_blue"]};">{v}</span></div>'
            for k, v in adaptive.items()
        ])
        adaptive_html = f'''
        <div class="strategy-box">
            <h4>🎯 Adaptive Portfolio Signals</h4>
            {adaptive_items}
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">💼</span>
            Portfolio Strategy
        </h2>

        <div class="strategy-grid">
            {entry_html}
            {tp_html}
            {adaptive_html}
        </div>

        {f'<div class="ai-analysis" style="margin-top: 20px;"><h4 style="color: {COLORS["accent_blue"]}; margin-bottom: 12px;">GPT Investment Recommendations</h4><div style="white-space: pre-wrap;">{gpt_recs}</div></div>' if gpt_recs else ''}
    </div>
    '''


# ============================================================================
# Technical Indicators Section
# ============================================================================

def _generate_technical_indicators_section(ai_report_data: Dict) -> str:
    indicators = ai_report_data.get('technical_indicators', {})

    if not indicators:
        return ''

    def format_value(val):
        if isinstance(val, float):
            return f'{val:.2f}'
        return str(val)

    indicator_items = ''
    indicator_map = {
        'vix': ('VIX', ''),
        'rsi_14': ('RSI (14)', ''),
        'macd': ('MACD', ''),
        'ma_50': ('MA 50', '$'),
        'ma_200': ('MA 200', '$'),
        'current_price': ('Current Price', '$'),
        'support_level': ('Support', '$'),
        'resistance_level': ('Resistance', '$'),
    }

    for key, (label, prefix) in indicator_map.items():
        if key in indicators:
            val = indicators[key]
            indicator_items += f'''
            <div class="indicator-item">
                <div class="indicator-label">{label}</div>
                <div class="indicator-value">{prefix}{format_value(val)}</div>
            </div>
            '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">📊</span>
            Technical Indicators
        </h2>
        <div class="indicator-grid">{indicator_items}</div>
    </div>
    '''


# ============================================================================
# AI Analysis Section
# ============================================================================

def _generate_ai_analysis_section(ai_report_data: Dict) -> str:
    claude_analysis = ai_report_data.get('claude_analysis', '')

    if not claude_analysis:
        return ''

    # Markdown을 HTML로 간단 변환
    html_content = claude_analysis
    html_content = html_content.replace('# ', '<h1>').replace('\n\n## ', '</h1>\n\n<h2>')
    html_content = html_content.replace('\n## ', '</h2>\n\n<h2>')
    html_content = html_content.replace('\n### ', '</h3>\n\n<h3>')
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    html_content = html_content.replace('\n\n', '</p><p>')
    html_content = html_content.replace('\n- ', '<br>• ')

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">🤖</span>
            AI Comprehensive Analysis (Claude)
        </h2>
        <div class="ai-analysis">{html_content}</div>
    </div>
    '''


# ============================================================================
# Detailed Signals Section (NEW)
# ============================================================================

def _generate_detailed_signals_section(integrated_data: Dict, signals_data: List[Dict] = None) -> str:
    """
    상세 시그널 섹션 생성 (action_guide + theory_note 포함)

    Args:
        integrated_data: 통합 데이터
        signals_data: 시그널 리스트 (외부에서 전달 가능)
    """
    # 시그널 데이터 가져오기
    signals = signals_data or integrated_data.get('detailed_signals', [])

    if not signals:
        return ''

    # 시그널 레벨별 카운트
    level_counts = {'CRITICAL': 0, 'ALERT': 0, 'WARNING': 0}
    for s in signals:
        level = s.get('level', 'WARNING').upper()
        if level in level_counts:
            level_counts[level] += 1

    # 요약 카드 생성
    summary_html = f'''
    <div class="signal-summary">
        <div class="signal-summary-card">
            <div class="signal-count" style="color: {COLORS['accent_red']};">{level_counts['CRITICAL']}</div>
            <div class="signal-count-label">Critical Signals</div>
        </div>
        <div class="signal-summary-card">
            <div class="signal-count" style="color: {COLORS['accent_yellow']};">{level_counts['ALERT']}</div>
            <div class="signal-count-label">Alert Signals</div>
        </div>
        <div class="signal-summary-card">
            <div class="signal-count" style="color: {COLORS['accent_blue']};">{level_counts['WARNING']}</div>
            <div class="signal-count-label">Warning Signals</div>
        </div>
        <div class="signal-summary-card">
            <div class="signal-count">{len(signals)}</div>
            <div class="signal-count-label">Total Signals</div>
        </div>
    </div>
    '''

    # 시그널 카드 생성 (CRITICAL > ALERT > WARNING 순)
    sorted_signals = sorted(signals, key=lambda x: {
        'CRITICAL': 0, 'ALERT': 1, 'WARNING': 2
    }.get(x.get('level', 'WARNING').upper(), 3))

    signal_cards = ''
    for s in sorted_signals[:20]:  # 최대 20개만 표시
        level = s.get('level', 'WARNING').upper()
        level_class = level.lower()
        level_badge_class = f'level-{level.lower()}'

        ticker = s.get('ticker', 'N/A')
        name = s.get('name', ticker)
        indicator = s.get('indicator', 'Unknown')
        value = s.get('value', 0)
        threshold = s.get('threshold', 0)
        z_score = s.get('z_score', 0)
        description = s.get('description', '')
        action_guide = s.get('action_guide', '')
        theory_note = s.get('theory_note', '')
        risk_prob = s.get('risk_prob', 0)

        # 값 포맷팅
        if isinstance(value, (int, float)):
            value_str = f'{value:.2f}' if abs(value) < 1000 else f'{value:,.0f}'
        else:
            value_str = str(value)

        if isinstance(threshold, (int, float)):
            threshold_str = f'{threshold:.2f}' if abs(threshold) < 1000 else f'{threshold:,.0f}'
        else:
            threshold_str = str(threshold)

        signal_cards += f'''
        <div class="signal-card {level_class}">
            <div class="signal-header">
                <div>
                    <span class="signal-ticker">{ticker}</span>
                    <span style="color: {COLORS['text_muted']}; margin-left: 8px;">{name}</span>
                </div>
                <span class="signal-level {level_badge_class}">{level}</span>
            </div>
            <div class="signal-indicator">{indicator}: {description}</div>
            <div class="signal-value">
                <div class="signal-value-item">
                    <div class="signal-value-label">Value</div>
                    <div class="signal-value-number">{value_str}</div>
                </div>
                <div class="signal-value-item">
                    <div class="signal-value-label">Threshold</div>
                    <div class="signal-value-number">{threshold_str}</div>
                </div>
                <div class="signal-value-item">
                    <div class="signal-value-label">Z-Score</div>
                    <div class="signal-value-number" style="color: {COLORS['accent_red'] if abs(z_score) > 2 else COLORS['text_primary']};">{z_score:.2f}</div>
                </div>
                {f'<div class="signal-value-item"><div class="signal-value-label">Risk Prob</div><div class="signal-value-number" style="color: {COLORS["accent_red"] if risk_prob > 0.5 else COLORS["text_primary"]};">{risk_prob:.1%}</div></div>' if risk_prob > 0 else ''}
            </div>
            {f'<div class="signal-guide"><div class="signal-guide-title">Action Guide</div><div class="signal-guide-text">{action_guide}</div></div>' if action_guide else ''}
            {f'<div class="signal-theory"><div class="signal-theory-title">Academic Reference</div><div class="signal-theory-text">{theory_note}</div></div>' if theory_note else ''}
        </div>
        '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">🔔</span>
            Detailed Market Signals
        </h2>

        <p style="color: {COLORS['text_secondary']}; margin-bottom: 20px;">
            기술적/통계적 지표 기반 시그널 탐지 결과입니다. 각 시그널에는 행동 가이드와 학술적 근거가 포함됩니다.
        </p>

        {summary_html}
        {signal_cards}

        {f'<div style="text-align: center; color: {COLORS["text_muted"]}; margin-top: 16px;">... and {len(signals) - 20} more signals</div>' if len(signals) > 20 else ''}
    </div>
    '''


# ============================================================================
# Asset Risk Metrics Section (NEW)
# ============================================================================

def _generate_asset_risk_metrics_section(integrated_data: Dict, metrics_data: Dict = None) -> str:
    """
    자산별 리스크 메트릭 섹션 생성

    Sharpe Ratio, Sortino Ratio, VaR 95%, CVaR 95%, Max Drawdown, Calmar Ratio
    """
    # 메트릭 데이터 가져오기
    metrics = metrics_data or integrated_data.get('asset_risk_metrics', {})

    if not metrics:
        return ''

    # metrics가 dict인 경우 (ticker -> metrics)
    if isinstance(metrics, dict):
        metrics_list = []
        for ticker, m in metrics.items():
            if isinstance(m, dict):
                m['ticker'] = ticker
                metrics_list.append(m)
        metrics = metrics_list

    if not metrics:
        return ''

    # 고위험/저위험 자산 분류
    high_risk = []
    low_risk = []
    for m in metrics:
        if isinstance(m, dict):
            sharpe = m.get('sharpe_ratio', 0)
            mdd = m.get('max_drawdown', 0)
            if sharpe < 0 or mdd < -0.15:
                high_risk.append(m.get('ticker', 'N/A'))
            elif sharpe > 1 and mdd > -0.1:
                low_risk.append(m.get('ticker', 'N/A'))

    # 요약 카드
    avg_sharpe = sum(m.get('sharpe_ratio', 0) for m in metrics if isinstance(m, dict)) / len(metrics) if metrics else 0
    avg_mdd = sum(m.get('max_drawdown', 0) for m in metrics if isinstance(m, dict)) / len(metrics) if metrics else 0

    summary_html = f'''
    <div class="cards-grid" style="margin-bottom: 20px;">
        <div class="card">
            <div class="card-header">
                <span class="card-title">Average Sharpe Ratio</span>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {COLORS['accent_green'] if avg_sharpe > 0.5 else COLORS['accent_red'] if avg_sharpe < 0 else COLORS['accent_yellow']};">
                {avg_sharpe:.2f}
            </div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-top: 8px;">
                {'Excellent' if avg_sharpe > 1 else 'Good' if avg_sharpe > 0.5 else 'Fair' if avg_sharpe > 0 else 'Poor'}
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <span class="card-title">Average Max Drawdown</span>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {COLORS['accent_red'] if avg_mdd < -0.15 else COLORS['accent_yellow'] if avg_mdd < -0.1 else COLORS['accent_green']};">
                {avg_mdd:.1%}
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <span class="card-title">High Risk Assets</span>
                <span class="card-badge badge-bearish">{len(high_risk)}</span>
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                {', '.join(high_risk[:5]) if high_risk else 'None'}
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <span class="card-title">Low Risk Assets</span>
                <span class="card-badge badge-bullish">{len(low_risk)}</span>
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                {', '.join(low_risk[:5]) if low_risk else 'None'}
            </div>
        </div>
    </div>
    '''

    # 테이블 행 생성
    table_rows = ''
    sorted_metrics = sorted(metrics, key=lambda x: x.get('sharpe_ratio', 0) if isinstance(x, dict) else 0, reverse=True)

    for m in sorted_metrics[:30]:  # 최대 30개
        if not isinstance(m, dict):
            continue

        ticker = m.get('ticker', 'N/A')
        name = m.get('name', ticker)
        sharpe = m.get('sharpe_ratio', 0)
        sortino = m.get('sortino_ratio', 0)
        var_95 = m.get('var_95', 0)
        cvar_95 = m.get('cvar_95', 0)
        max_dd = m.get('max_drawdown', 0)
        calmar = m.get('calmar_ratio', 0)

        # Sharpe 색상
        sharpe_color = COLORS['accent_green'] if sharpe > 1 else COLORS['accent_yellow'] if sharpe > 0 else COLORS['accent_red']
        # Sortino 색상
        sortino_color = COLORS['accent_green'] if sortino > 1.5 else COLORS['accent_yellow'] if sortino > 0 else COLORS['accent_red']
        # MDD 색상
        mdd_color = COLORS['accent_red'] if max_dd < -0.15 else COLORS['accent_yellow'] if max_dd < -0.1 else COLORS['accent_green']

        # MDD 바 (0~50% 범위로 정규화)
        mdd_width = min(abs(max_dd) * 200, 100)

        table_rows += f'''
        <tr>
            <td><strong>{ticker}</strong><br><span style="color: {COLORS['text_muted']}; font-size: 0.8rem;">{name[:20]}</span></td>
            <td style="color: {sharpe_color}; font-weight: 600;">{sharpe:.2f}</td>
            <td style="color: {sortino_color}; font-weight: 600;">{sortino:.2f}</td>
            <td style="color: {COLORS['accent_red']};">{var_95:.2%}</td>
            <td style="color: {COLORS['accent_red']};">{cvar_95:.2%}</td>
            <td>
                <div class="metric-bar">
                    <span style="color: {mdd_color}; font-weight: 600;">{max_dd:.1%}</span>
                    <div style="flex: 1; background: {COLORS['bg_primary']}; border-radius: 3px; height: 6px;">
                        <div class="metric-bar-fill" style="width: {mdd_width}%; background: {mdd_color};"></div>
                    </div>
                </div>
            </td>
            <td style="color: {COLORS['accent_blue']};">{calmar:.2f}</td>
        </tr>
        '''

    # 범례
    legend_html = f'''
    <div class="metrics-legend">
        <div class="metrics-legend-item">
            <div class="legend-dot" style="background: {COLORS['accent_green']};"></div>
            <span>Sharpe > 1.0 (우수)</span>
        </div>
        <div class="metrics-legend-item">
            <div class="legend-dot" style="background: {COLORS['accent_yellow']};"></div>
            <span>0 < Sharpe < 1.0 (보통)</span>
        </div>
        <div class="metrics-legend-item">
            <div class="legend-dot" style="background: {COLORS['accent_red']};"></div>
            <span>Sharpe < 0 (저조)</span>
        </div>
    </div>
    '''

    return f'''
    <div class="section">
        <h2 class="section-title">
            <span class="section-icon">📉</span>
            Asset Risk Metrics
        </h2>

        <p style="color: {COLORS['text_secondary']}; margin-bottom: 20px;">
            자산별 위험조정수익률 및 리스크 메트릭입니다. Sharpe(1966), Sortino & Price(1994), VaR(RiskMetrics 1994), CVaR(Artzner 1999) 방법론 적용.
        </p>

        {summary_html}

        <div style="overflow-x: auto;">
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Sharpe<br><span style="font-weight: 400; font-size: 0.75rem;">위험조정수익률</span></th>
                        <th>Sortino<br><span style="font-weight: 400; font-size: 0.75rem;">하방위험조정</span></th>
                        <th>VaR 95%<br><span style="font-weight: 400; font-size: 0.75rem;">최대손실추정</span></th>
                        <th>CVaR 95%<br><span style="font-weight: 400; font-size: 0.75rem;">꼬리위험</span></th>
                        <th>Max DD<br><span style="font-weight: 400; font-size: 0.75rem;">최대낙폭</span></th>
                        <th>Calmar<br><span style="font-weight: 400; font-size: 0.75rem;">MDD대비수익</span></th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        {legend_html}
    </div>
    '''


# ============================================================================
# Footer
# ============================================================================

def _generate_footer() -> str:
    return f'''
    <div class="report-footer">
        <p><strong>EIMAS - Economic Intelligence Multi-Agent System</strong></p>
        <p style="margin-top: 8px;">
            This report is generated by AI agents analyzing macroeconomic data, market indicators,
            and real-time news. The recommendations should be used as reference only and do not
            constitute financial advice.
        </p>
        <div class="data-sources">
            Data Sources: FRED (Federal Reserve), Yahoo Finance, Perplexity AI, OpenAI GPT-4o, Anthropic Claude
        </div>
    </div>
    '''


# ============================================================================
# CLI 실행
# ============================================================================

def generate_signals_and_metrics(market_data: Dict = None) -> tuple:
    """
    시그널 및 리스크 메트릭 생성

    Args:
        market_data: 시장 데이터 (ticker -> DataFrame with Close, Volume)

    Returns:
        (signals_list, metrics_dict)
    """
    signals_data = []
    metrics_data = {}

    if not market_data:
        return signals_data, metrics_data

    # Signal Analyzer 사용
    if SignalAnalyzer is not None:
        try:
            analyzer = SignalAnalyzer()
            signals_result = analyzer.analyze_all(market_data)
            # analyze_all returns List[Signal]
            if signals_result:
                signals_data = [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals_result]
        except Exception as e:
            print(f"Warning: Signal analysis failed: {e}")

    # Risk Metrics Calculator 사용
    if AssetRiskCalculator is not None:
        try:
            calculator = AssetRiskCalculator()
            # analyze_all returns RiskMetricsResult with metrics dict
            result = calculator.analyze_all(market_data)
            if result and hasattr(result, 'metrics'):
                for ticker, m in result.metrics.items():
                    metrics_data[ticker] = m.to_dict() if hasattr(m, 'to_dict') else m
            elif isinstance(result, dict):
                metrics_data = result
        except Exception as e:
            print(f"Warning: Risk metrics failed: {e}")

    return signals_data, metrics_data


def _flatten_yfinance_columns(df):
    """
    yfinance의 multi-level column을 단일 레벨로 변환

    yfinance download()는 단일 티커도 MultiIndex 컬럼을 반환할 수 있음
    """
    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', 'SPY') -> 'Close'
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def main():
    """CLI 실행"""
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Generate EIMAS Portfolio Report')
    parser.add_argument('--integrated', type=str, help='Path to integrated_*.json')
    parser.add_argument('--ai-report', type=str, help='Path to ai_report_*.json')
    parser.add_argument('--output', type=str, default='outputs/reports/portfolio_report.html')
    parser.add_argument('--latest', action='store_true', help='Use latest files automatically')
    parser.add_argument('--with-signals', action='store_true', help='Generate fresh signals from market data')
    parser.add_argument('--with-metrics', action='store_true', help='Generate fresh risk metrics from market data')
    parser.add_argument('--standalone', action='store_true', help='Run independently without main.py (generates all data)')
    args = parser.parse_args()

    # standalone 모드면 자동으로 signals/metrics 생성
    if args.standalone:
        args.with_signals = True
        args.with_metrics = True

    outputs_dir = Path(__file__).parent.parent / 'outputs'

    # standalone 모드에서는 파일 검색 건너뛰기
    if not args.standalone:
        # 최신 파일 자동 선택
        if args.latest or (not args.integrated):
            integrated_files = sorted(glob.glob(str(outputs_dir / 'integrated_*.json')))
            if integrated_files:
                args.integrated = integrated_files[-1]
                print(f"Using integrated: {args.integrated}")

        if args.latest or (not args.ai_report):
            ai_report_files = sorted(glob.glob(str(outputs_dir / 'ai_report_*.json')))
            if ai_report_files:
                args.ai_report = ai_report_files[-1]
                print(f"Using AI report: {args.ai_report}")

    # standalone 모드: integrated 파일 없이 기본 데이터 생성
    if not args.integrated:
        if args.standalone:
            print("📝 Running in standalone mode (no integrated_*.json needed)")
            integrated_data = {
                'timestamp': datetime.now().isoformat(),
                'final_recommendation': 'ANALYZING',
                'confidence': 0.5,
                'risk_score': 50.0,
                'risk_level': 'MEDIUM',
                'regime': {'regime': 'Unknown', 'trend': 'Unknown', 'volatility': 'Unknown', 'confidence': 0},
                'fred_summary': {'rrp': 0, 'tga': 0, 'net_liquidity': 0, 'fed_funds': 0, 'treasury_10y': 0, 'spread_10y2y': 0},
                'full_mode_position': 'N/A',
                'reference_mode_position': 'N/A',
                'modes_agree': True,
                'events_detected': [],
            }
        else:
            print("Error: No integrated_*.json file found. Use --standalone to run without it.")
            return
    else:
        # 데이터 로드
        with open(args.integrated, 'r', encoding='utf-8') as f:
            integrated_data = json.load(f)

    ai_report_data = {}
    if args.ai_report and os.path.exists(args.ai_report):
        with open(args.ai_report, 'r', encoding='utf-8') as f:
            ai_report_data = json.load(f)

    # 시그널/메트릭 데이터
    signals_data = integrated_data.get('detailed_signals', [])
    metrics_data = integrated_data.get('asset_risk_metrics', {})

    # 새로 생성 요청 시 (yfinance 데이터 필요)
    if args.with_signals or args.with_metrics:
        try:
            import yfinance as yf
            print("\n📊 Fetching market data for signal/metrics generation...")

            # 기본 티커 목록
            tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'SLV', 'USO',
                      'XLK', 'XLF', 'XLV', 'XLE', '^VIX', 'BTC-USD', 'ETH-USD']

            market_data = {}
            for ticker in tickers:
                try:
                    df = yf.download(ticker, period='90d', progress=False, auto_adjust=True)
                    if not df.empty:
                        # yfinance multi-level column 처리
                        df = _flatten_yfinance_columns(df)
                        market_data[ticker] = df
                except Exception as e:
                    pass

            if market_data:
                print(f"   Downloaded {len(market_data)} tickers")
                new_signals, new_metrics = generate_signals_and_metrics(market_data)
                if args.with_signals and new_signals:
                    signals_data = new_signals
                    print(f"   Generated {len(signals_data)} signals")
                if args.with_metrics and new_metrics:
                    metrics_data = new_metrics
                    print(f"   Generated metrics for {len(metrics_data)} assets")
        except ImportError:
            print("Warning: yfinance not installed. Using existing data.")
        except Exception as e:
            import traceback
            print(f"Warning: Failed to generate fresh data: {e}")
            traceback.print_exc()

    # 리포트 생성
    output_path = args.output
    html = generate_portfolio_report(
        integrated_data,
        ai_report_data,
        output_path,
        signals_data=signals_data,
        metrics_data=metrics_data
    )

    print(f"\n✅ Report generated: {output_path}")
    print(f"   Size: {len(html) / 1024:.1f} KB")
    print(f"\n   Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
