#!/usr/bin/env python3
"""
Dashboard Generator Module
===========================
EIMAS 분석 결과를 인터랙티브 HTML 대시보드로 시각화.

기능:
- 자산군별 위험 현황
- 레짐 분석 (BULL/BEAR/TRANSITION/CRISIS)
- LASSO 예측 결과
- 멀티에이전트 토론 결과
- Critical Path 분석
- 위험 메트릭
- 거시경제 지표
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 로거 설정
logger = logging.getLogger('eimas.dashboard')


# ============================================================================
# 상수 및 설정
# ============================================================================

ASSET_CATEGORIES = {
    '주식': {
        'tickers': ['SPY', 'QQQ', 'IWM', 'EEM', 'XLF', 'XLK', 'XLY', 'XLP', 'XLRE', 'VNQ', 'RSP'],
        'icon': '📈',
        'color': '#3b82f6'
    },
    '채권': {
        'tickers': ['TLT', 'HYG', 'LQD', 'SHY', 'IEF', 'TIP'],
        'icon': '📊',
        'color': '#8b5cf6'
    },
    '원자재': {
        'tickers': ['GLD', 'GC=F', 'SI=F', 'SLV', 'CL=F', 'HG=F', 'ZW=F', 'ZC=F', 'NG=F', 'DBA', 'DBC'],
        'icon': '🏭',
        'color': '#f59e0b'
    },
    '환율': {
        'tickers': ['DX-Y.NYB', 'USDJPY=X', 'USDKRW=X', 'USDCNY=X', 'EURUSD=X', 'GBPUSD=X'],
        'icon': '💱',
        'color': '#10b981'
    },
    '암호화폐': {
        'tickers': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'],
        'icon': '🪙',
        'color': '#ec4899'
    }
}

# 다크 테마 색상
THEME_COLORS = {
    'dark': {
        'bg': '#1a1a2e',
        'card_bg': '#16213e',
        'text': '#e0e0e0',
        'text_muted': '#9ca3af',
        'accent': '#4a90d9',
        'positive': '#22c55e',
        'negative': '#ef4444',
        'neutral': '#f59e0b',
        'border': 'rgba(255, 255, 255, 0.1)'
    },
    'light': {
        'bg': '#ffffff',
        'card_bg': '#f5f5f5',
        'text': '#333333',
        'text_muted': '#666666',
        'accent': '#3b82f6',
        'positive': '#16a34a',
        'negative': '#dc2626',
        'neutral': '#d97706',
        'border': 'rgba(0, 0, 0, 0.1)'
    }
}


# ============================================================================
# 메인 대시보드 생성 함수
# ============================================================================

def generate_dashboard(
    signals: List[Dict] = None,
    summary: str = "",
    interpretations: List[Dict] = None,
    news: List[Dict] = None,
    regime_data: Dict = None,
    crypto_panel: Dict = None,
    risk_data: Dict = None,
    critical_path_data: Dict = None,
    risk_metrics: Dict = None,
    macro_indicators: Dict = None,
    llm_summary: str = "",
    agent_opinions: List[Any] = None,
    consensus: Any = None,
    conflicts: List[Any] = None,
    forecast_results: List[Any] = None,
    theme: str = 'dark',
    language: str = 'ko'
) -> str:
    """
    전체 대시보드 HTML 생성
    
    Args:
        signals: 이상 신호 목록
        summary: 요약 텍스트
        interpretations: AI 해석
        news: 뉴스 데이터
        regime_data: 레짐 정보
        crypto_panel: 암호화폐 패널
        risk_data: ML 기반 위험 확률
        critical_path_data: Critical Path 분석
        risk_metrics: 위험조정수익률
        macro_indicators: 거시경제 지표
        llm_summary: LLM 요약
        agent_opinions: 에이전트 의견 목록
        consensus: 합의 결과
        conflicts: 충돌 목록
        forecast_results: LASSO 예측 결과
        theme: 테마 ('dark' / 'light')
        language: 언어 ('ko' / 'en')
        
    Returns:
        HTML 문자열
    """
    # 기본값 설정
    signals = signals or []
    interpretations = interpretations or []
    news = news or []
    regime_data = regime_data or {}
    risk_metrics = risk_metrics or {}
    macro_indicators = macro_indicators or {}
    agent_opinions = agent_opinions or []
    conflicts = conflicts or []
    forecast_results = forecast_results or []
    
    colors = THEME_COLORS.get(theme, THEME_COLORS['dark'])
    timestamp = datetime.now().isoformat()
    
    # HTML 시작
    html = _generate_html_header(timestamp, theme, colors)
    
    # 요약 섹션
    html += _generate_summary_section(summary, len(signals), colors)
    
    # 레짐 섹션
    if regime_data:
        html += _generate_regime_section(regime_data, colors)
    
    # LASSO 예측 결과
    if forecast_results:
        html += generate_lasso_section(forecast_results, colors)
    
    # 멀티에이전트 섹션
    if agent_opinions:
        html += generate_multi_agent_section(agent_opinions, consensus, conflicts, colors)
    
    # 자산군별 위험 현황
    if signals:
        html += generate_asset_risk_section(signals, colors)
    
    # 위험 메트릭
    if risk_metrics:
        html += _generate_risk_metrics_section(risk_metrics, colors)
    
    # 거시경제 지표
    if macro_indicators:
        html += _generate_macro_section(macro_indicators, colors)
    
    # LLM 요약
    if llm_summary:
        html += _generate_llm_summary_section(llm_summary, colors)
    
    # HTML 종료
    html += _generate_html_footer()
    
    return html


# ============================================================================
# 멀티에이전트 섹션 함수
# ============================================================================

def generate_multi_agent_section(
    opinions: List[Any],
    consensus: Any = None,
    conflicts: List[Any] = None,
    colors: Dict = None
) -> str:
    """
    멀티에이전트 토론 결과를 HTML로 시각화
    
    Args:
        opinions: AgentOpinion 객체 또는 딕셔너리 목록
        consensus: Consensus 객체 또는 딕셔너리
        conflicts: Conflict 객체 또는 딕셔너리 목록
        colors: 테마 색상 (optional)
        
    Returns:
        HTML 문자열
        
    UI 레이아웃:
    ┌─────────────────────────────────────────────────────────────┐
    │  🤖 Multi-Agent Analysis                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │ AnalysisAgent│  │ForecastAgent │  │StrategyAgent│       │
    │  │   BEARISH    │  │    HOLD      │  │   CAUTIOUS  │       │
    │  │  conf: 0.75  │  │  conf: 0.68  │  │  conf: 0.72 │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘      │
    ├─────────────────────────────────────────────────────────────┤
    │  📊 Consensus: CAUTIOUS HOLD (Agreement: 78%)              │
    │  ⚠️  Conflicts: rate_magnitude (ForecastAgent vs Strategy) │
    └─────────────────────────────────────────────────────────────┘
    """
    colors = colors or THEME_COLORS['dark']
    conflicts = conflicts or []
    
    # 에이전트 카드 생성
    cards_html = ""
    for opinion in opinions:
        # 딕셔너리 또는 객체 처리
        if isinstance(opinion, dict):
            agent_id = opinion.get('agent_role', opinion.get('agent_id', 'Unknown'))
            position = opinion.get('position', 'N/A')
            confidence = opinion.get('confidence', 0.0)
            reasoning = opinion.get('reasoning', '')[:100]
        else:
            agent_id = getattr(opinion, 'agent_role', 'Unknown')
            if hasattr(agent_id, 'value'):
                agent_id = agent_id.value
            position = getattr(opinion, 'position', 'N/A')
            confidence = getattr(opinion, 'confidence', 0.0)
            reasoning = getattr(opinion, 'reasoning', '')[:100] if hasattr(opinion, 'reasoning') else ''
        
        # 포지션에 따른 색상 결정
        position_upper = str(position).upper()
        if any(x in position_upper for x in ['UP', 'HIKE', 'BULLISH', 'BUY', 'LONG']):
            border_color = colors['positive']
        elif any(x in position_upper for x in ['DOWN', 'CUT', 'BEARISH', 'SELL', 'SHORT']):
            border_color = colors['negative']
        else:
            border_color = colors['neutral']
        
        cards_html += f'''
        <div class="agent-card" style="border-left: 4px solid {border_color};">
            <div class="agent-name">{agent_id}</div>
            <div class="agent-position" style="color: {border_color};">{position}</div>
            <div class="agent-confidence">conf: {confidence:.2f}</div>
            {f'<div class="agent-reasoning">{reasoning}...</div>' if reasoning else ''}
        </div>
        '''
    
    # 합의 박스 생성
    consensus_html = ""
    if consensus:
        if isinstance(consensus, dict):
            final_position = consensus.get('final_position', 'N/A')
            agreement_score = consensus.get('confidence', consensus.get('agreement_score', 0.0))
        else:
            final_position = getattr(consensus, 'final_position', 'N/A')
            agreement_score = getattr(consensus, 'confidence', 0.0)
        
        consensus_html = f'''
        <div class="consensus-box">
            <span class="consensus-icon">📊</span>
            <span class="consensus-text">
                Consensus: <strong>{final_position}</strong> 
                (Agreement: {agreement_score:.0%})
            </span>
        </div>
        '''
    
    # 충돌 목록 생성
    conflicts_html = ""
    if conflicts:
        conflicts_html = '<ul class="conflict-list">'
        for conflict in conflicts:
            if isinstance(conflict, dict):
                topic = conflict.get('topic', 'Unknown')
                agents = conflict.get('agents', [])
                agent_a = agents[0] if len(agents) > 0 else 'Agent A'
                agent_b = agents[1] if len(agents) > 1 else 'Agent B'
            else:
                topic = getattr(conflict, 'topic', 'Unknown')
                agents = getattr(conflict, 'agents', [])
                agent_a = agents[0].value if hasattr(agents[0], 'value') else str(agents[0]) if agents else 'Agent A'
                agent_b = agents[1].value if len(agents) > 1 and hasattr(agents[1], 'value') else str(agents[1]) if len(agents) > 1 else 'Agent B'
            
            conflicts_html += f'<li>⚠️ {topic}: {agent_a} vs {agent_b}</li>'
        conflicts_html += '</ul>'
    
    return f'''
    <div class="section" id="multi-agent-section">
        <h2 class="section-title">🤖 Multi-Agent Analysis</h2>
        <div class="agent-cards">
            {cards_html}
        </div>
        {consensus_html}
        {conflicts_html}
    </div>
    '''


# ============================================================================
# LASSO 결과 섹션 함수
# ============================================================================

def generate_lasso_section(
    results: List[Any],
    colors: Dict = None,
    diagnostics: Dict = None
) -> str:
    """
    LASSO 분석 결과를 HTML로 시각화
    
    Args:
        results: ForecastResult 객체 또는 딕셔너리 목록
        colors: 테마 색상 (optional)
        diagnostics: 진단 정보 딕셔너리 (optional)
        
    Returns:
        HTML 문자열
        
    UI 레이아웃:
    ┌─────────────────────────────────────────────────────────────┐
    │  📈 LASSO Fed Rate Forecast                                 │
    ├─────────────────────────────────────────────────────────────┤
    │  Horizon      │ R²    │ Selected │ Top Variables           │
    │  ─────────────┼───────┼──────────┼─────────────────────────│
    │  VeryShort    │ 0.00  │ 1        │ d_Breakeven5Y           │
    │  Short        │ 0.37  │ 7        │ d_HighYield_Rate, ...   │
    │  Long         │ 0.64  │ 28       │ d_Baa_Yield, ...        │
    ├─────────────────────────────────────────────────────────────┤
    │  [Horizontal Bar Chart: Top 10 Coefficients]               │
    │  ████████████████████ d_Baa_Yield (+2.09)                  │
    │  ██████████████████   d_Spread_Baa (-1.66)                 │
    └─────────────────────────────────────────────────────────────┘
    """
    colors = colors or THEME_COLORS['dark']
    diagnostics = diagnostics or {}
    
    # 문제 진단 섹션 생성
    issues_html = ""
    issues = []
    
    # 결과가 없거나 모두 n_observations가 0인 경우
    if not results:
        issues.append("❌ 분석 결과 없음: LASSO 모델이 실행되지 않았습니다.")
    else:
        total_obs = 0
        for result in results:
            if isinstance(result, dict):
                n_obs = result.get('n_observations', 0)
            else:
                n_obs = getattr(result, 'n_observations', 0)
            total_obs += n_obs
        
        if total_obs == 0:
            issues.append("❌ 관측치 없음: 모든 horizon에서 n_observations = 0")
    
    # diagnostics에서 문제 추출
    if diagnostics:
        if diagnostics.get('common_dates', 0) == 0:
            issues.append(f"❌ 공통 날짜 없음: CME 데이터와 시장 데이터의 날짜가 겹치지 않음")
        elif diagnostics.get('common_dates', 0) < 30:
            issues.append(f"⚠️ 공통 날짜 부족: {diagnostics.get('common_dates')}개 (최소 30개 권장)")
        
        if not diagnostics.get('has_d_exp_rate', False):
            issues.append("❌ 종속변수 누락: d_Exp_Rate가 데이터에 없음")
        
        if diagnostics.get('feature_count', 0) < 5:
            issues.append(f"⚠️ 설명변수 부족: {diagnostics.get('feature_count', 0)}개 (최소 5개 권장)")
        
        if diagnostics.get('days_to_meeting_missing', False):
            issues.append("❌ days_to_meeting 누락: FOMC 일정 데이터 없음")
        
        # 추가 디버그 정보
        if diagnostics.get('market_data_rows', 0) == 0:
            issues.append("❌ 시장 데이터 없음: market_data가 비어있음")
        
        if diagnostics.get('cme_data_rows', 0) == 0:
            issues.append("❌ CME 데이터 없음: CME 패널 데이터 로드 실패")
        
        # 날짜 범위 정보
        if diagnostics.get('market_date_range'):
            issues.append(f"📅 시장 데이터 기간: {diagnostics['market_date_range']}")
        if diagnostics.get('cme_date_range'):
            issues.append(f"📅 CME 데이터 기간: {diagnostics['cme_date_range']}")
    
    if issues:
        issues_items = ''.join([f'<li>{issue}</li>' for issue in issues])
        issues_html = f'''
        <div class="diagnostics-box" style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        ">
            <h3 style="color: #e74c3c; margin-top: 0;">🔍 문제 진단 (Diagnostics)</h3>
            <ul style="color: {colors['text']}; margin: 0; padding-left: 20px;">
                {issues_items}
            </ul>
        </div>
        '''
    
    # 테이블 행 생성
    rows_html = ""
    for result in results:
        if isinstance(result, dict):
            horizon = result.get('horizon', 'Unknown')
            r_squared = result.get('r_squared', 0.0)
            selected_vars = result.get('selected_variables', [])
            n_selected = len(selected_vars)
            coefficients = result.get('coefficients', {})
            n_observations = result.get('n_observations', 0)
        else:
            horizon = getattr(result, 'horizon', 'Unknown')
            r_squared = getattr(result, 'r_squared', 0.0)
            selected_vars = getattr(result, 'selected_variables', [])
            n_selected = len(selected_vars)
            coefficients = getattr(result, 'coefficients', {})
            n_observations = getattr(result, 'n_observations', 0)
        
        # 상위 변수 (최대 3개)
        top_vars = selected_vars[:3]
        top_vars_str = ', '.join(top_vars) if top_vars else 'None'
        if len(selected_vars) > 3:
            top_vars_str += '...'
        
        # R² 색상
        if r_squared > 0.5:
            r2_color = colors['positive']
        elif r_squared > 0.2:
            r2_color = colors['neutral']
        else:
            r2_color = colors['text_muted']
        
        # n_observations 색상
        if n_observations == 0:
            n_obs_color = colors['negative']
            n_obs_warning = ' ⚠️'
        elif n_observations < 30:
            n_obs_color = colors['neutral']
            n_obs_warning = ''
        else:
            n_obs_color = colors['positive']
            n_obs_warning = ''
        
        rows_html += f'''
        <tr>
            <td>{horizon}</td>
            <td style="color: {n_obs_color};">{n_observations}{n_obs_warning}</td>
            <td style="color: {r2_color};">{r_squared:.4f}</td>
            <td>{n_selected}</td>
            <td>{top_vars_str}</td>
        </tr>
        '''
    
    # Long horizon 차트 데이터 준비
    chart_html = ""
    long_result = results[2] if len(results) > 2 else (results[-1] if results else None)
    
    if long_result:
        if isinstance(long_result, dict):
            coefficients = long_result.get('coefficients', {})
        else:
            coefficients = getattr(long_result, 'coefficients', {})
        
        if coefficients:
            # 절대값 기준 정렬하여 상위 10개
            sorted_coefs = sorted(
                coefficients.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            
            labels = [item[0] for item in sorted_coefs]
            values = [item[1] for item in sorted_coefs]
            bar_colors = [colors['positive'] if v > 0 else colors['negative'] for v in values]
            
            # JSON 문자열로 변환
            labels_json = json.dumps(labels)
            values_json = json.dumps(values)
            colors_json = json.dumps(bar_colors)
            
            chart_html = f'''
            <div class="chart-container" style="height: 350px; margin-top: 25px;">
                <canvas id="lassoCoefChart"></canvas>
            </div>
            <script>
                (function() {{
                    const ctx = document.getElementById('lassoCoefChart').getContext('2d');
                    new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: {labels_json},
                            datasets: [{{
                                label: 'Coefficient',
                                data: {values_json},
                                backgroundColor: {colors_json},
                                borderColor: {colors_json},
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            indexAxis: 'y',
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }},
                                title: {{
                                    display: true,
                                    text: 'Top 10 LASSO Coefficients (Long Horizon)',
                                    color: '{colors["text"]}',
                                    font: {{ size: 14 }}
                                }}
                            }},
                            scales: {{
                                x: {{
                                    grid: {{ color: '{colors["border"]}' }},
                                    ticks: {{ color: '{colors["text_muted"]}' }}
                                }},
                                y: {{
                                    grid: {{ display: false }},
                                    ticks: {{ color: '{colors["text"]}' }}
                                }}
                            }}
                        }}
                    }});
                }})();
            </script>
            '''
    
    return f'''
    <div class="section" id="lasso-section">
        <h2 class="section-title">📈 LASSO Fed Rate Forecast</h2>
        {issues_html}
        <table class="lasso-summary">
            <thead>
                <tr>
                    <th>Horizon</th>
                    <th>Obs</th>
                    <th>R²</th>
                    <th>Selected</th>
                    <th>Top Variables</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        {chart_html}
    </div>
    '''


# ============================================================================
# 자산군별 위험 현황 섹션
# ============================================================================

def generate_asset_risk_section(signals: List[Dict], colors: Dict = None) -> str:
    """자산군별 위험 현황 섹션 HTML 생성"""
    colors = colors or THEME_COLORS['dark']
    
    # 카테고리별 신호 그룹화
    category_signals = {cat: {'critical': 0, 'alert': 0, 'warning': 0, 'signals': []} 
                        for cat in ASSET_CATEGORIES.keys()}
    
    for signal in signals:
        ticker = signal.get('ticker', '')
        level = signal.get('level', 'WARNING').upper()
        
        for cat_name, cat_info in ASSET_CATEGORIES.items():
            if ticker in cat_info['tickers']:
                if level == 'CRITICAL':
                    category_signals[cat_name]['critical'] += 1
                elif level == 'ALERT':
                    category_signals[cat_name]['alert'] += 1
                else:
                    category_signals[cat_name]['warning'] += 1
                category_signals[cat_name]['signals'].append(signal)
                break
    
    # 카드 HTML 생성
    cards_html = ""
    for cat_name, cat_info in ASSET_CATEGORIES.items():
        stats = category_signals[cat_name]
        total = stats['critical'] + stats['alert'] + stats['warning']
        
        if stats['critical'] > 0:
            border_color = colors['negative']
            status = 'CRITICAL'
        elif stats['alert'] > 0:
            border_color = '#f97316'
            status = 'ALERT'
        elif stats['warning'] > 0:
            border_color = colors['neutral']
            status = 'WARNING'
        else:
            border_color = colors['positive']
            status = 'STABLE'
        
        cards_html += f'''
        <div class="asset-card" style="border-left: 4px solid {border_color};">
            <div class="asset-header">
                <span class="asset-icon">{cat_info['icon']}</span>
                <span class="asset-name">{cat_name}</span>
            </div>
            <div class="asset-status" style="color: {border_color};">{status}</div>
            <div class="asset-count">{total} signals</div>
        </div>
        '''
    
    return f'''
    <div class="section" id="asset-risk-section">
        <h2 class="section-title">⚠️ 자산군별 위험 현황</h2>
        <div class="asset-cards">
            {cards_html}
        </div>
    </div>
    '''


# ============================================================================
# 헬퍼 함수들
# ============================================================================

def _generate_html_header(timestamp: str, theme: str, colors: Dict) -> str:
    """HTML 헤더 생성"""
    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIMAS Dashboard - {timestamp[:10]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {colors['bg']};
            color: {colors['text']};
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid {colors['accent']};
            margin-bottom: 30px;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            color: {colors['accent']};
            margin-bottom: 10px;
        }}
        
        .timestamp {{ color: {colors['text_muted']}; font-size: 0.9rem; }}
        
        .section {{
            background: {colors['card_bg']};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .section-title {{
            font-size: 1.4rem;
            color: {colors['accent']};
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .agent-cards, .asset-cards {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .agent-card, .asset-card {{
            flex: 1;
            min-width: 200px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        
        .agent-name, .asset-name {{ font-weight: bold; margin-bottom: 8px; }}
        .agent-position {{ font-size: 1.2rem; margin-bottom: 5px; }}
        .agent-confidence {{ font-size: 0.85rem; color: {colors['text_muted']}; }}
        .agent-reasoning {{ font-size: 0.8rem; color: {colors['text_muted']}; margin-top: 8px; }}
        
        .asset-header {{ display: flex; align-items: center; gap: 8px; justify-content: center; }}
        .asset-icon {{ font-size: 1.5rem; }}
        .asset-status {{ font-size: 1.1rem; font-weight: bold; margin: 8px 0; }}
        .asset-count {{ font-size: 0.9rem; color: {colors['text_muted']}; }}
        
        .consensus-box {{
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid {colors['positive']};
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .conflict-list {{
            list-style: none;
            margin-top: 15px;
        }}
        
        .conflict-list li {{
            padding: 8px 0;
            border-bottom: 1px solid {colors['border']};
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid {colors['border']};
        }}
        
        th {{
            background: rgba(74, 144, 217, 0.2);
            font-weight: 600;
        }}
        
        .chart-container {{
            position: relative;
            margin-top: 20px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px 0;
            color: {colors['text_muted']};
            font-size: 0.85rem;
            border-top: 1px solid {colors['border']};
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 EIMAS Dashboard</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
'''


def _generate_summary_section(summary: str, signal_count: int, colors: Dict) -> str:
    """요약 섹션 생성"""
    return f'''
        <div class="section">
            <h2 class="section-title">📊 Summary</h2>
            <div class="card-grid">
                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px;">
                    <div style="font-size: 0.9rem; color: {colors['text_muted']};">Total Signals</div>
                    <div style="font-size: 2rem; font-weight: bold;">{signal_count}</div>
                </div>
            </div>
            {f'<p style="margin-top: 15px; color: {colors["text_muted"]};">{summary}</p>' if summary else ''}
        </div>
    '''


def _generate_regime_section(regime_data: Dict, colors: Dict) -> str:
    """레짐 분석 섹션"""
    regime = regime_data.get('current_regime', 'UNKNOWN')
    probability = regime_data.get('probability', 0.0)
    
    regime_colors = {
        'BULL': colors['positive'],
        'BEAR': colors['negative'],
        'TRANSITION': colors['neutral'],
        'CRISIS': '#dc2626',
        'UNKNOWN': colors['text_muted']
    }
    color = regime_colors.get(regime, colors['text_muted'])
    
    return f'''
    <div class="section">
        <h2 class="section-title">📈 Regime Analysis</h2>
        <div class="card-grid">
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px; border-left: 4px solid {color};">
                <div style="font-size: 0.9rem; color: {colors['text_muted']};">Current Regime</div>
                <div style="font-size: 2rem; font-weight: bold; color: {color};">{regime}</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px;">
                <div style="font-size: 0.9rem; color: {colors['text_muted']};">Confidence</div>
                <div style="font-size: 2rem; font-weight: bold;">{probability:.1%}</div>
            </div>
        </div>
    </div>
    '''


def _generate_risk_metrics_section(metrics: Dict, colors: Dict) -> str:
    """위험 메트릭 섹션"""
    cards = ""
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            cards += f'''
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px;">
                <div style="font-size: 0.9rem; color: {colors['text_muted']};">{name}</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{formatted}</div>
            </div>
            '''
    
    return f'''
    <div class="section">
        <h2 class="section-title">⚠️ Risk Metrics</h2>
        <div class="card-grid">{cards}</div>
    </div>
    '''


def _generate_macro_section(indicators: Dict, colors: Dict) -> str:
    """거시경제 지표 섹션"""
    rows = ""
    for name, value in indicators.items():
        if isinstance(value, (int, float)):
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            rows += f"<tr><td>{name}</td><td>{formatted}</td></tr>"
    
    return f'''
    <div class="section">
        <h2 class="section-title">🌍 Macro Indicators</h2>
        <table>
            <thead><tr><th>Indicator</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    '''


def _generate_llm_summary_section(summary: str, colors: Dict) -> str:
    """LLM 요약 섹션"""
    return f'''
    <div class="section">
        <h2 class="section-title">🤖 AI Analysis</h2>
        <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; line-height: 1.8;">
            {summary}
        </div>
    </div>
    '''


def _generate_html_footer() -> str:
    """HTML 푸터"""
    return '''
        <footer>
            <p>Generated by EIMAS (Economic Intelligence Multi-Agent System)</p>
            <p>© 2025 - Dashboard v1.0</p>
        </footer>
    </div>
</body>
</html>'''


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_position_color(position: str, colors: Dict) -> str:
    """포지션에 따른 색상 반환"""
    position_upper = str(position).upper()
    if any(x in position_upper for x in ['UP', 'HIKE', 'BULLISH', 'BUY', 'LONG']):
        return colors['positive']
    elif any(x in position_upper for x in ['DOWN', 'CUT', 'BEARISH', 'SELL', 'SHORT']):
        return colors['negative']
    else:
        return colors['neutral']


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=== Dashboard Generator Test ===\n")
    
    # 테스트 데이터
    test_forecast_results = [
        {
            'horizon': 'VeryShort',
            'r_squared': 0.02,
            'selected_variables': ['d_Breakeven5Y'],
            'coefficients': {'d_Breakeven5Y': 0.15}
        },
        {
            'horizon': 'Short',
            'r_squared': 0.35,
            'selected_variables': ['d_Spread_Baa', 'd_HighYield_Rate', 'Ret_VIX'],
            'coefficients': {'d_Spread_Baa': -0.42, 'd_HighYield_Rate': 0.35, 'Ret_VIX': 0.28}
        },
        {
            'horizon': 'Long',
            'r_squared': 0.64,
            'selected_variables': ['d_Baa_Yield', 'd_Spread_Baa', 'Ret_Dollar_Idx', 'd_Breakeven5Y'],
            'coefficients': {
                'd_Baa_Yield': 2.09,
                'd_Spread_Baa': -1.66,
                'Ret_Dollar_Idx': 1.04,
                'd_Breakeven5Y': 0.85
            }
        }
    ]
    
    test_opinions = [
        {'agent_role': 'analysis', 'position': 'BEARISH', 'confidence': 0.75},
        {'agent_role': 'forecast', 'position': 'HOLD', 'confidence': 0.68},
        {'agent_role': 'strategy', 'position': 'CAUTIOUS', 'confidence': 0.72}
    ]
    
    test_consensus = {'final_position': 'CAUTIOUS HOLD', 'confidence': 0.78}
    
    test_conflicts = [{'topic': 'rate_magnitude', 'agents': ['forecast', 'strategy']}]
    
    # 대시보드 생성
    html = generate_dashboard(
        signals=[{'ticker': 'SPY', 'level': 'ALERT'}],
        summary="Test dashboard generation",
        regime_data={'current_regime': 'TRANSITION', 'probability': 0.72},
        forecast_results=test_forecast_results,
        agent_opinions=test_opinions,
        consensus=test_consensus,
        conflicts=test_conflicts
    )
    
    # 파일 저장
    output_path = 'outputs/dashboards/test_dashboard.html'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Dashboard generated: {output_path}")
    print(f"✓ Size: {len(html) / 1024:.1f} KB")
    print("\n=== Test Completed ===")

