#!/usr/bin/env python3
"""
Market Anomaly Detector - Dashboard Generator
==============================================
하드코딩된 HTML 템플릿으로 안정적인 대시보드 생성
+ Crypto Panel 지원
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

# 자산군 카테고리 정의
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


def generate_asset_risk_section(signals: List[Dict]) -> str:
    """자산군별 위험 현황 섹션 HTML 생성"""
    
    # 카테고리별 신호 그룹화
    category_signals = {cat: {'critical': 0, 'alert': 0, 'warning': 0, 'signals': []} 
                        for cat in ASSET_CATEGORIES.keys()}
    
    for signal in signals:
        ticker = signal.get('ticker', '')
        level = signal.get('level', 'WARNING').upper()
        
        # 티커가 어느 카테고리에 속하는지 확인
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
    
    # HTML 생성
    cards_html = ""
    for cat_name, cat_info in ASSET_CATEGORIES.items():
        stats = category_signals[cat_name]
        total = stats['critical'] + stats['alert'] + stats['warning']
        
        # 위험 수준 결정
        if stats['critical'] > 0:
            risk_level = 'CRITICAL'
            border_color = '#ef4444'
        elif stats['alert'] > 0:
            risk_level = 'ALERT'
            border_color = '#f97316'
        elif stats['warning'] > 0:
            risk_level = 'WARNING'
            border_color = '#eab308'
        else:
            risk_level = 'STABLE'
            border_color = '#22c55e'
        
        # 주요 신호 (최대 3개)
        key_signals_html = ""
        for sig in stats['signals'][:3]:
            sig_level = sig.get('level', 'WARNING')
            sig_color = {'CRITICAL': '#ef4444', 'ALERT': '#f97316', 'WARNING': '#eab308'}.get(sig_level, '#22c55e')
            key_signals_html += f"""
            <div class="key-signal-item" style="border-left: 3px solid {sig_color}; padding-left: 8px; margin: 4px 0;">
                <span style="font-weight: 600;">{sig.get('name', sig.get('ticker', ''))}</span>
                <span style="color: #9ca3af; font-size: 0.8em;"> - {sig.get('indicator', '')}</span>
            </div>
            """
        
        if not key_signals_html:
            key_signals_html = '<div style="color: #22c55e; font-size: 0.9em;">✓ 이상 신호 없음</div>'
        
        cards_html += f"""
        <div class="asset-category-card" style="
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 1rem;
            border-left: 4px solid {border_color};
        ">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5em;">{cat_info['icon']}</span>
                <h4 style="margin: 0; color: #f1f5f9;">{cat_name}</h4>
                <span style="
                    margin-left: auto;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.75em;
                    font-weight: 600;
                    background: {border_color}20;
                    color: {border_color};
                ">{risk_level}</span>
            </div>
            <div class="risk-counts" style="display: flex; gap: 12px; margin-bottom: 0.75rem; font-size: 0.85em;">
                <span style="color: #ef4444;">● CRITICAL: {stats['critical']}</span>
                <span style="color: #f97316;">● ALERT: {stats['alert']}</span>
                <span style="color: #eab308;">● WARNING: {stats['warning']}</span>
            </div>
            <div class="key-signals" style="font-size: 0.85em;">
                {key_signals_html}
            </div>
        </div>
        """
    
    return f"""
    <div class="asset-risk-section">
        <h3 style="color: #f1f5f9; margin-bottom: 1rem;">📊 자산군별 위험 현황</h3>
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        ">
            {cards_html}
        </div>
    </div>
    """


def generate_regime_display(regime_data: Dict) -> str:
    """개선된 레짐 표시 HTML 생성"""
    
    current_regime = regime_data.get('current_regime', 'TRANSITION')
    confidence = regime_data.get('regime_confidence', 50.0)
    transition_prob = regime_data.get('transition_probability', 0.0)
    
    # 레짐별 색상 및 아이콘
    regime_styles = {
        'BULL': {'color': '#22c55e', 'icon': '🟢', 'bg': 'rgba(34, 197, 94, 0.2)'},
        'BEAR': {'color': '#ef4444', 'icon': '🔴', 'bg': 'rgba(239, 68, 68, 0.2)'},
        'TRANSITION': {'color': '#eab308', 'icon': '🟡', 'bg': 'rgba(234, 179, 8, 0.2)'},
        'CRISIS': {'color': '#dc2626', 'icon': '⚠️', 'bg': 'rgba(220, 38, 38, 0.2)'}
    }
    
    style = regime_styles.get(current_regime, regime_styles['TRANSITION'])
    
    # 확신도 게이지 바
    confidence_bar = f"""
    <div style="margin-top: 0.5rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #9ca3af;">
            <span>레짐 확신도</span>
            <span>{confidence:.1f}%</span>
        </div>
        <div style="background: #374151; border-radius: 4px; height: 8px; margin-top: 4px;">
            <div style="
                background: {style['color']};
                width: {min(confidence, 100)}%;
                height: 100%;
                border-radius: 4px;
                transition: width 0.3s;
            "></div>
        </div>
    </div>
    """
    
    # 전환 확률 표시
    transition_html = ""
    if transition_prob > 20:
        transition_color = '#ef4444' if transition_prob > 50 else '#eab308'
        transition_html = f"""
        <div style="margin-top: 0.75rem; padding: 8px; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9ca3af; font-size: 0.85em;">⚡ 전환 확률</span>
                <span style="color: {transition_color}; font-weight: 600;">{transition_prob:.1f}%</span>
            </div>
        </div>
        """
    
    return f"""
    <div class="regime-display" style="
        background: {style['bg']};
        border: 1px solid {style['color']}40;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    ">
        <div style="font-size: 3em; margin-bottom: 0.5rem;">{style['icon']}</div>
        <div style="
            font-size: 1.5em;
            font-weight: 700;
            color: {style['color']};
            margin-bottom: 0.25rem;
        ">{current_regime}</div>
        <div style="color: #9ca3af; font-size: 0.9em;">시장 국면</div>
        {confidence_bar}
        {transition_html}
    </div>
    """


def generate_crypto_panel_html(
    crypto_signals: List[Dict],
    crypto_news: List[Dict],
    crypto_collection_status: Dict
) -> str:
    """암호화폐 전용 패널 HTML 생성"""
    
    # 수집 상태 HTML
    status = crypto_collection_status or {}
    successful = status.get('successful', 0)
    failed = status.get('failed', 0)
    total = status.get('total_tickers', 0)
    fallback_used = status.get('fallback_used_count', 0)
    
    status_color = '#22c55e' if failed == 0 else '#eab308' if failed < total else '#ef4444'
    status_icon = '✅' if failed == 0 else '⚠️' if failed < total else '❌'
    
    status_html = f"""
    <div class="crypto-status">
        <span class="status-icon">{status_icon}</span>
        <span class="status-text">데이터 수집: {successful}/{total} 성공</span>
        {f'<span class="fallback-badge">Fallback: {fallback_used}</span>' if fallback_used > 0 else ''}
    </div>
    """
    
    # 수집 상세 상태
    details_html = ""
    for ticker, detail in status.get('tickers', {}).items():
        icon = '✅' if detail.get('success') else '❌'
        source = detail.get('source', 'N/A')
        name = detail.get('name', ticker)
        details_html += f"""
        <div class="crypto-status-item">
            <span>{icon}</span>
            <span class="ticker">{ticker}</span>
            <span class="name">{name}</span>
            <span class="source">{source}</span>
        </div>
        """
    
    # Crypto 신호 테이블
    signal_rows = ""
    for s in crypto_signals:
        level = s.get('level', 'WARNING')
        level_color = {
            'CRITICAL': '#ef4444',
            'ALERT': '#f97316',
            'WARNING': '#eab308'
        }.get(level, '#22c55e')
        
        signal_rows += f"""
        <tr>
            <td><strong>{s.get('name', s.get('ticker', ''))}</strong></td>
            <td>{s.get('indicator', '')}</td>
            <td><span class="level-badge" style="background: {level_color};">{level}</span></td>
            <td>{s.get('description', '')}</td>
        </tr>
        """
    
    # Crypto 뉴스 HTML
    news_html = ""
    for n in crypto_news[:5]:  # 최대 5개
        ticker = n.get('ticker', '')
        headline = n.get('headline', n.get('news', ''))[:200]
        summary = n.get('summary', '')[:300]
        
        news_html += f"""
        <div class="crypto-news-item">
            <div class="news-ticker">{ticker}</div>
            <div class="news-headline">{headline}{'...' if len(headline) >= 200 else ''}</div>
            {f'<div class="news-summary">{summary}</div>' if summary else ''}
        </div>
        """
    
    return f"""
    <div class="crypto-panel">
        <h3>🪙 암호화폐 패널 (Crypto Panel)</h3>
        
        <!-- 수집 상태 -->
        <div class="crypto-collection-status">
            <h4>📊 데이터 수집 상태</h4>
            {status_html}
            <div class="crypto-status-details">
                {details_html if details_html else '<p style="color: #9ca3af;">상태 정보 없음</p>'}
            </div>
        </div>
        
        <!-- Crypto 신호 -->
        <div class="crypto-signals">
            <h4>⚡ Crypto 신호</h4>
            {f'''<table class="crypto-table">
                <thead>
                    <tr>
                        <th>자산</th>
                        <th>지표</th>
                        <th>레벨</th>
                        <th>설명</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_rows}
                </tbody>
            </table>''' if signal_rows else '<p style="color: #9ca3af;">감지된 암호화폐 신호 없음</p>'}
        </div>
        
        <!-- Crypto 뉴스 -->
        <div class="crypto-news">
            <h4>📰 Crypto 뉴스</h4>
            {news_html if news_html else '<p style="color: #9ca3af;">관련 뉴스 없음</p>'}
        </div>
    </div>
    """


def _generate_signal_news_section(signal_news: List[Dict]) -> str:
    """
    Signal News 패널 HTML 생성
    
    Args:
        signal_news: [{ "signal": {...}, "news": "뉴스 분석 텍스트" }, ...]
    
    Returns:
        HTML 문자열
    """
    if not signal_news:
        return ""
    
    # ALERT 레벨 이상의 신호만 필터링
    alert_signals = [
        item for item in signal_news 
        if item.get('signal', {}).get('level') in ['CRITICAL', 'ALERT']
    ]
    
    if not alert_signals:
        return ""
    
    news_items_html = ""
    for idx, item in enumerate(alert_signals[:5]):  # 최대 5개
        signal = item.get('signal', {})
        news_text = item.get('news', '')
        
        if not news_text:
            continue
        
        # 마크다운을 HTML로 변환
        # 헤더 변환
        news_html = re.sub(r'^##\s+(.+)$', r'<h4>\1</h4>', news_text, flags=re.MULTILINE)
        # 볼드 변환 (**text** -> <strong>text</strong>)
        news_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', news_html)
        # 줄바꿈 변환
        news_html = news_html.replace('\n', '<br>')
        
        ticker = signal.get('ticker', 'N/A')
        name = signal.get('name', ticker)
        level = signal.get('level', 'ALERT')
        description = signal.get('description', '')
        
        level_color = {
            'CRITICAL': '#ef4444',
            'ALERT': '#f97316',
            'WARNING': '#eab308'
        }.get(level, '#9ca3af')
        
        news_items_html += f"""
        <div class="signal-news-item">
            <div class="signal-news-header" onclick="toggleNews({idx})">
                <div class="signal-news-info">
                    <span class="signal-news-ticker">{ticker}</span>
                    <span class="signal-news-name">{name}</span>
                    <span class="signal-news-level" style="background: {level_color};">{level}</span>
                </div>
                <div class="signal-news-toggle">▼</div>
            </div>
            <div class="signal-news-content" id="news-content-{idx}" style="display: none;">
                <div class="signal-news-description">{description}</div>
                <div class="signal-news-text">{news_html}</div>
            </div>
        </div>
        """
    
    return f"""
    <div class="signal-news-section">
        <h3>📰 신호별 뉴스 배경</h3>
        <div class="signal-news-list">
            {news_items_html if news_items_html else '<p style="color: #9ca3af;">뉴스 정보 없음</p>'}
        </div>
    </div>
    """


def _generate_spillover_section(spillover_result: Dict) -> str:
    """
    Spillover Analysis 상세 HTML 생성
    
    Args:
        spillover_result: {
            'active_paths': [...],
            'risk_score': float,
            'primary_risk_source': str,
            'expected_impacts': {...}
        }
    
    Returns:
        HTML 문자열
    """
    if not spillover_result:
        return ""
    
    active_paths = spillover_result.get('active_paths', [])
    risk_score = spillover_result.get('risk_score', 0)
    primary_source = spillover_result.get('primary_risk_source', 'N/A')
    expected_impacts = spillover_result.get('expected_impacts', {})
    
    # 위험 점수 색상
    if risk_score >= 70:
        risk_color = '#ef4444'
    elif risk_score >= 50:
        risk_color = '#f97316'
    elif risk_score >= 30:
        risk_color = '#eab308'
    else:
        risk_color = '#22c55e'
    
    # 활성 경로 HTML
    paths_html = ""
    if active_paths:
        for path in active_paths[:10]:  # 최대 10개
            source = path.get('source', 'N/A')
            target = path.get('target', 'N/A')
            strength = path.get('strength', 0)
            category = path.get('category', 'unknown')
            
            # 강도에 따른 색상
            if strength >= 0.7:
                strength_color = '#ef4444'
            elif strength >= 0.5:
                strength_color = '#f97316'
            elif strength >= 0.3:
                strength_color = '#eab308'
            else:
                strength_color = '#22c55e'
            
            category_names = {
                'liquidity': '유동성',
                'credit': '신용',
                'volatility': '변동성',
                'sentiment': '센티먼트',
                'correlation': '상관관계'
            }
            category_name = category_names.get(category, category)
            
            paths_html += f"""
            <div class="spillover-path-item">
                <div class="spillover-path-header">
                    <span class="spillover-source">{source}</span>
                    <span class="spillover-arrow">→</span>
                    <span class="spillover-target">{target}</span>
                    <span class="spillover-category">{category_name}</span>
                </div>
                <div class="spillover-path-strength">
                    <div class="spillover-strength-bar" style="width: {strength * 100}%; background: {strength_color};"></div>
                    <span class="spillover-strength-value">{strength:.2f}</span>
                </div>
            </div>
            """
    else:
        paths_html = '<p style="color: #9ca3af;">현재 활성화된 충격 전이 경로 없음</p>'
    
    # 예상 영향 HTML
    impacts_html = ""
    if expected_impacts:
        for asset, impact in list(expected_impacts.items())[:5]:
            impact_value = impact if isinstance(impact, (int, float)) else 0
            impacts_html += f"""
            <div class="spillover-impact-item">
                <span class="impact-asset">{asset}</span>
                <span class="impact-value" style="color: {'#ef4444' if impact_value >= 0.5 else '#f97316' if impact_value >= 0.3 else '#eab308'};">{impact_value:.2f}</span>
            </div>
            """
    
    return f"""
    <div class="spillover-detail-section">
        <h4>🔄 충격 전이 분석 (Spillover Analysis)</h4>
        <div class="spillover-summary">
            <div class="spillover-metric">
                <span class="spillover-metric-label">스필오버 위험 점수</span>
                <span class="spillover-metric-value" style="color: {risk_color};">{risk_score:.1f}</span>
            </div>
            <div class="spillover-metric">
                <span class="spillover-metric-label">주요 위험 소스</span>
                <span class="spillover-metric-value">{primary_source}</span>
            </div>
        </div>
        <div class="spillover-paths">
            <h5>활성 전이 경로</h5>
            {paths_html}
        </div>
        {f'''<div class="spillover-impacts">
            <h5>예상 영향</h5>
            {impacts_html}
        </div>''' if impacts_html else ''}
    </div>
    """


def _generate_ma_status_section(ma_status: Dict) -> str:
    """
    MA Status 상세 정보 HTML 생성
    
    Args:
        ma_status: {
            'ma_5': float,
            'ma_20': float,
            'ma_120': float,
            'price_vs_ma20': float,  # %
            'price_vs_ma120': float,  # %
            'ma20_slope': float,
            'ma120_slope': float
        }
    
    Returns:
        HTML 문자열
    """
    if not ma_status:
        return ""
    
    ma_5 = ma_status.get('ma_5', 0)
    ma_20 = ma_status.get('ma_20', 0)
    ma_120 = ma_status.get('ma_120', 0)
    price_vs_ma20 = ma_status.get('price_vs_ma20', 0)
    price_vs_ma120 = ma_status.get('price_vs_ma120', 0)
    ma20_slope = ma_status.get('ma20_slope', 0)
    ma120_slope = ma_status.get('ma120_slope', 0)
    
    # 기울기에 따른 화살표
    def get_slope_arrow(slope):
        if slope > 0.01:
            return '↗'
        elif slope < -0.01:
            return '↘'
        else:
            return '→'
    
    def get_slope_color(slope):
        if slope > 0.01:
            return '#22c55e'
        elif slope < -0.01:
            return '#ef4444'
        else:
            return '#9ca3af'
    
    ma20_arrow = get_slope_arrow(ma20_slope)
    ma120_arrow = get_slope_arrow(ma120_slope)
    ma20_color = get_slope_color(ma20_slope)
    ma120_color = get_slope_color(ma120_slope)
    
    # 이격도 색상 (과열/과냉 판단)
    def get_deviation_color(deviation):
        if abs(deviation) > 10:
            return '#ef4444'  # 과열/과냉
        elif abs(deviation) > 5:
            return '#f97316'  # 주의
        else:
            return '#22c55e'  # 정상
    
    ma20_dev_color = get_deviation_color(price_vs_ma20)
    ma120_dev_color = get_deviation_color(price_vs_ma120)
    
    return f"""
    <div class="ma-status-section">
        <h4>📊 이동평균 상태 (MA Status)</h4>
        <div class="ma-values">
            <div class="ma-value-item">
                <span class="ma-label">MA5</span>
                <span class="ma-value">{ma_5:.2f}</span>
            </div>
            <div class="ma-value-item">
                <span class="ma-label">MA20</span>
                <span class="ma-value">{ma_20:.2f}</span>
                <span class="ma-slope" style="color: {ma20_color};">{ma20_arrow}</span>
            </div>
            <div class="ma-value-item">
                <span class="ma-label">MA120</span>
                <span class="ma-value">{ma_120:.2f}</span>
                <span class="ma-slope" style="color: {ma120_color};">{ma120_arrow}</span>
            </div>
        </div>
        <div class="ma-deviations">
            <div class="ma-deviation-item">
                <div class="ma-deviation-label">현재가 vs MA20</div>
                <div class="ma-deviation-bar-container">
                    <div class="ma-deviation-bar" style="width: {min(abs(price_vs_ma20), 20) * 5}%; background: {ma20_dev_color}; margin-left: {'50%' if price_vs_ma20 >= 0 else f'{50 - abs(price_vs_ma20) * 2.5}%'}"></div>
                </div>
                <div class="ma-deviation-value" style="color: {ma20_dev_color};">{price_vs_ma20:+.1f}%</div>
            </div>
            <div class="ma-deviation-item">
                <div class="ma-deviation-label">현재가 vs MA120</div>
                <div class="ma-deviation-bar-container">
                    <div class="ma-deviation-bar" style="width: {min(abs(price_vs_ma120), 20) * 5}%; background: {ma120_dev_color}; margin-left: {'50%' if price_vs_ma120 >= 0 else f'{50 - abs(price_vs_ma120) * 2.5}%'}"></div>
                </div>
                <div class="ma-deviation-value" style="color: {ma120_dev_color};">{price_vs_ma120:+.1f}%</div>
            </div>
        </div>
    </div>
    """


def _generate_risk_summary_section(summary: str) -> str:
    """
    Risk Summary 요약문 HTML 생성
    
    Args:
        summary: 위험 요약 텍스트
    
    Returns:
        HTML 문자열
    """
    if not summary or summary == "위험 모델 미적용":
        return ""
    
    # 숫자 강조 (퍼센트, 개수 등)
    import re
    # 숫자 패턴 찾아서 강조
    summary_html = re.sub(
        r'(\d+(?:\.\d+)?%)',
        r'<strong style="color: #60a5fa;">\1</strong>',
        summary
    )
    summary_html = re.sub(
        r'(\d+)\s*개',
        r'<strong style="color: #60a5fa;">\1개</strong>',
        summary_html
    )
    
    # 줄바꿈 처리
    summary_html = summary_html.replace('\n', '<br>')
    
    return f"""
    <div class="risk-summary-section">
        <h4>📋 위험 요약</h4>
        <div class="risk-summary-content">{summary_html}</div>
    </div>
    """


def _generate_markov_regime_section(markov_analysis: Dict) -> str:
    """
    Markov Switching Regime 분석 섹션 HTML 생성
    
    Args:
        markov_analysis: {ticker: {transition_matrix, expected_duration, next_regime_prob, regime_history}, ...}
    
    Returns:
        HTML 문자열
    """
    if not markov_analysis:
        return ""
    
    # 주요 자산 (SPY, QQQ)만 표시
    main_tickers = ['SPY', 'QQQ']
    displayed_tickers = [t for t in main_tickers if t in markov_analysis]
    
    if not displayed_tickers:
        # 주요 자산이 없으면 첫 번째 자산 사용
        displayed_tickers = [list(markov_analysis.keys())[0]] if markov_analysis else []
    
    sections_html = ""
    for ticker in displayed_tickers[:2]:  # 최대 2개
        analysis = markov_analysis[ticker]
        transition_matrix = analysis.get('transition_matrix', [])
        expected_duration = analysis.get('expected_duration', {})
        next_regime_prob = analysis.get('next_regime_prob', {})
        regime_history = analysis.get('regime_history', [])
        
        # 전이확률 행렬 HTML
        transition_html = ""
        if transition_matrix:
            n_regimes = len(transition_matrix)
            regime_names = ['BULL', 'NEUTRAL', 'BEAR'][:n_regimes] if n_regimes == 3 else ['BULL', 'BEAR']
            
            transition_html = "<table class='transition-matrix-table'>"
            transition_html += "<thead><tr><th>From \\ To</th>"
            for name in regime_names:
                transition_html += f"<th>{name}</th>"
            transition_html += "</tr></thead><tbody>"
            
            for i, row in enumerate(transition_matrix):
                transition_html += f"<tr><td><strong>{regime_names[i]}</strong></td>"
                for j, prob in enumerate(row):
                    # 확률에 따른 색상
                    if prob >= 0.8:
                        color = '#22c55e'  # 녹색 (높은 확률)
                    elif prob >= 0.5:
                        color = '#eab308'  # 노란색
                    else:
                        color = '#ef4444'  # 빨간색 (낮은 확률)
                    transition_html += f"<td style='color: {color}; font-weight: 600;'>{prob:.3f}</td>"
                transition_html += "</tr>"
            transition_html += "</tbody></table>"
        
        # 예상 지속 기간 HTML
        duration_html = ""
        for regime, duration in expected_duration.items():
            duration_str = f"{duration}일" if isinstance(duration, (int, float)) else str(duration)
            duration_html += f"<div class='duration-item'><span class='duration-regime'>{regime}</span><span class='duration-value'>{duration_str}</span></div>"
        
        # 다음 regime 전환 확률 HTML
        next_prob_html = ""
        for regime, prob in next_regime_prob.items():
            prob_pct = prob * 100
            if prob_pct >= 50:
                color = '#22c55e'
            elif prob_pct >= 30:
                color = '#eab308'
            else:
                color = '#9ca3af'
            next_prob_html += f"<div class='next-prob-item'><span class='next-prob-regime'>{regime}</span><span class='next-prob-value' style='color: {color};'>{prob_pct:.1f}%</span></div>"
        
        # Regime 확률 시계열 차트 데이터 (JSON)
        chart_data_json = '[]'
        if regime_history and len(regime_history) > 0:
            # 최근 60일만 표시
            recent_history = regime_history[-60:] if len(regime_history) > 60 else regime_history
            chart_data = []
            for i in range(n_regimes):
                regime_name = regime_names[i] if i < len(regime_names) else f'Regime_{i+1}'
                probs = []
                dates = []
                for h in recent_history:
                    # regime_history는 dict 리스트이므로 키로 접근
                    prob_key = f'Regime_{i+1}'
                    if prob_key in h:
                        probs.append(float(h[prob_key]))
                    else:
                        probs.append(0.0)
                    # 날짜 추출
                    if 'index' in h:
                        dates.append(str(h['index']))
                    elif 'date' in h:
                        dates.append(str(h['date']))
                
                if probs:
                    chart_data.append({
                        'label': regime_name,
                        'data': probs,
                        'dates': dates if dates else [str(i) for i in range(len(probs))]
                    })
            
            chart_data_json = json.dumps(chart_data, ensure_ascii=False) if chart_data else '[]'
        
        sections_html += f"""
        <div class="markov-regime-card">
            <h4>📊 {ticker} - Markov Switching 분석</h4>
            
            <div class="markov-transition-section">
                <h5>전이확률 행렬 (Transition Matrix)</h5>
                <div class="markov-note">각 행은 현재 regime에서 다른 regime으로 전환할 확률</div>
                {transition_html if transition_html else '<p style="color: #9ca3af;">데이터 없음</p>'}
            </div>
            
            <div class="markov-metrics-grid">
                <div class="markov-metric-card">
                    <h5>예상 지속 기간</h5>
                    <div class="duration-list">
                        {duration_html if duration_html else '<p style="color: #9ca3af;">데이터 없음</p>'}
                    </div>
                </div>
                
                <div class="markov-metric-card">
                    <h5>다음 Regime 전환 확률</h5>
                    <div class="next-prob-list">
                        {next_prob_html if next_prob_html else '<p style="color: #9ca3af;">데이터 없음</p>'}
                    </div>
                </div>
            </div>
            
            <div class="markov-chart-section">
                <h5>Regime 확률 시계열 (최근 60일)</h5>
                <div class="chart-container">
                    <canvas id="markov-chart-{ticker}"></canvas>
                </div>
            </div>
        </div>
        """
    
    return f"""
    <div class="markov-regime-section">
        <h3>🔬 확률적 Regime 분석 (Markov Switching Model)</h3>
        <div class="markov-note-intro">
            Hamilton(1989) Markov Switching 모델 기반 분석. 시장이 여러 regime 사이를 확률적으로 전환한다고 가정하여 각 시점의 regime 확률을 추정합니다.
        </div>
        {sections_html if sections_html else '<p style="color: #9ca3af;">Markov 분석 데이터 없음</p>'}
    </div>
    """


def _generate_markov_charts_js(markov_analysis: Dict) -> str:
    """
    Markov Switching Regime 확률 시계열 차트를 위한 JavaScript 코드 생성
    
    Args:
        markov_analysis: {ticker: {regime_history, transition_matrix}, ...}
    
    Returns:
        JavaScript 코드 문자열
    """
    if not markov_analysis:
        return ""
    
    js_code = ""
    
    # 주요 자산 (SPY, QQQ)만 처리
    main_tickers = ['SPY', 'QQQ']
    displayed_tickers = [t for t in main_tickers if t in markov_analysis]
    
    if not displayed_tickers:
        displayed_tickers = [list(markov_analysis.keys())[0]] if markov_analysis else []
    
    for ticker in displayed_tickers[:2]:  # 최대 2개
        analysis = markov_analysis[ticker]
        regime_history = analysis.get('regime_history', [])
        transition_matrix = analysis.get('transition_matrix', [])
        
        if not regime_history or not transition_matrix:
            continue
        
        n_regimes = len(transition_matrix)
        regime_names = ['BULL', 'NEUTRAL', 'BEAR'][:n_regimes] if n_regimes == 3 else ['BULL', 'BEAR']
        regime_colors = ['#22c55e', '#eab308', '#ef4444'][:n_regimes] if n_regimes == 3 else ['#22c55e', '#ef4444']
        
        # 최근 60일만 표시
        recent_history = regime_history[-60:] if len(regime_history) > 60 else regime_history
        
        # 차트 데이터 준비
        datasets = []
        labels = []
        
        for i in range(n_regimes):
            regime_name = regime_names[i] if i < len(regime_names) else f'Regime_{i+1}'
            probs = []
            dates = []
            
            for h in recent_history:
                prob_key = f'Regime_{i+1}'
                if prob_key in h:
                    probs.append(float(h[prob_key]) * 100)  # 퍼센트로 변환
                else:
                    probs.append(0.0)
                
                # 날짜 추출
                if 'index' in h:
                    dates.append(str(h['index']))
                elif 'date' in h:
                    dates.append(str(h['date']))
                else:
                    dates.append('')
            
            if probs:
                datasets.append({
                    'label': regime_name,
                    'data': probs,
                    'borderColor': regime_colors[i] if i < len(regime_colors) else '#9ca3af',
                    'backgroundColor': regime_colors[i] + '40' if i < len(regime_colors) else '#9ca3af40',
                    'fill': True,
                    'tension': 0.4
                })
                
                if not labels:
                    labels = dates if dates else [str(i) for i in range(len(probs))]
        
        # JavaScript 코드 생성
        chart_id = f'markov-chart-{ticker}'
        datasets_json = json.dumps(datasets, ensure_ascii=False)
        labels_json = json.dumps(labels, ensure_ascii=False)
        
        js_code += f"""
            try {{
                const markovCtx_{ticker.replace('-', '_')} = document.getElementById('{chart_id}');
                if (!markovCtx_{ticker.replace('-', '_')}) {{
                    console.warn('{chart_id} 캔버스를 찾을 수 없습니다.');
                }} else {{
                    const markovChart_{ticker.replace('-', '_')} = new Chart(markovCtx_{ticker.replace('-', '_')}.getContext('2d'), {{
                        type: 'line',
                        data: {{
                            labels: {labels_json},
                            datasets: {datasets_json}
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    position: 'top',
                                    labels: {{ color: '#e4e4e7' }}
                                }},
                                tooltip: {{
                                    mode: 'index',
                                    intersect: false
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: {{
                                        color: '#9ca3af',
                                        callback: function(value) {{
                                            return value + '%';
                                        }}
                                    }},
                                    grid: {{ color: 'rgba(255,255,255,0.1)' }}
                                }},
                                x: {{
                                    ticks: {{ color: '#9ca3af' }},
                                    grid: {{ display: false }}
                                }}
                            }}
                        }}
                    }});
                }}
            }} catch (error) {{
                console.error('Markov chart 생성 실패 ({ticker}):', error);
                const canvas = document.getElementById('{chart_id}');
                if (canvas) {{
                    const container = canvas.parentElement;
                    const errorDiv = document.createElement('div');
                    errorDiv.style.cssText = 'padding: 20px; text-align: center; color: #fca5a5;';
                    errorDiv.innerHTML = '<p>차트 로드 실패</p>';
                    container.appendChild(errorDiv);
                }}
            }}
        """
    
    return js_code


def _generate_risk_metrics_section(risk_metrics: Dict[str, Dict]) -> str:
    """
    위험조정수익률 지표 섹션 HTML 생성
    
    Args:
        risk_metrics: {ticker: {sharpe_ratio, sortino_ratio, var_95, cvar_95, max_drawdown, calmar_ratio}, ...}
    
    Returns:
        HTML 문자열
    """
    if not risk_metrics:
        return ""
    
    # 최대 10개 자산만 표시 (Sharpe Ratio 기준 정렬)
    sorted_tickers = sorted(
        risk_metrics.items(),
        key=lambda x: x[1].get('sharpe_ratio', 0),
        reverse=True
    )[:10]
    
    if not sorted_tickers:
        return ""
    
    rows_html = ""
    for ticker, metrics in sorted_tickers:
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        var_95 = metrics.get('var_95', 0)
        cvar_95 = metrics.get('cvar_95', 0)
        max_dd = metrics.get('max_drawdown', 0)
        calmar = metrics.get('calmar_ratio', 0)
        
        # Sharpe Ratio 색상 코딩
        if sharpe > 1:
            sharpe_color = '#22c55e'  # 녹색
        elif sharpe > 0:
            sharpe_color = '#eab308'  # 노란색
        else:
            sharpe_color = '#ef4444'  # 빨간색
        
        # Max Drawdown 색상 (음수이므로 절댓값으로 판단)
        max_dd_abs = abs(max_dd)
        if max_dd_abs > 0.3:
            max_dd_color = '#ef4444'  # 빨간색 (30% 이상 하락)
        elif max_dd_abs > 0.2:
            max_dd_color = '#f97316'  # 주황색 (20-30%)
        elif max_dd_abs > 0.1:
            max_dd_color = '#eab308'  # 노란색 (10-20%)
        else:
            max_dd_color = '#22c55e'  # 녹색 (10% 미만)
        
        rows_html += f"""
        <tr>
            <td><strong>{ticker}</strong></td>
            <td style="color: {sharpe_color}; font-weight: 600;">{sharpe:.2f}</td>
            <td style="color: {sharpe_color if sortino > 1 else '#eab308' if sortino > 0 else '#ef4444'}; font-weight: 600;">{sortino:.2f}</td>
            <td style="color: {'#ef4444' if var_95 < -0.05 else '#f97316' if var_95 < -0.03 else '#eab308'}; font-weight: 600;">{var_95*100:.2f}%</td>
            <td style="color: {'#ef4444' if cvar_95 < -0.05 else '#f97316' if cvar_95 < -0.03 else '#eab308'}; font-weight: 600;">{cvar_95*100:.2f}%</td>
            <td style="color: {max_dd_color}; font-weight: 600;">{max_dd*100:.2f}%</td>
            <td style="color: {'#22c55e' if calmar > 1 else '#eab308' if calmar > 0 else '#ef4444'}; font-weight: 600;">{calmar:.2f}</td>
        </tr>
        """
    
    return f"""
    <div class="risk-metrics-section">
        <h4>📈 투자 성과 지표 (Risk-Adjusted Return Metrics)</h4>
        <div class="risk-metrics-note" style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 12px;">
            기관투자자들이 사용하는 위험조정수익률 지표입니다. Sharpe > 1: 양호, Sortino: 하방위험 고려, VaR/CVaR: 최대 예상 손실, Max DD: 최대 하락폭, Calmar: Drawdown 대비 수익률
        </div>
        <div class="risk-metrics-table-container">
            <table class="risk-metrics-table">
                <thead>
                    <tr>
                        <th>자산</th>
                        <th>Sharpe Ratio</th>
                        <th>Sortino Ratio</th>
                        <th>VaR (95%)</th>
                        <th>CVaR (95%)</th>
                        <th>Max Drawdown</th>
                        <th>Calmar Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else '<tr><td colspan="7" style="text-align: center; color: #9ca3af;">데이터 없음</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
    """


def _generate_macro_environment_section(macro_indicators: Dict) -> str:
    """
    거시경제 환경 섹션 HTML 생성
    
    Args:
        macro_indicators: {
            'yield_curve_slope': float,
            'yield_curve_status': str,
            'credit_spread_change': float,
            'ted_spread': float,
            'interpretation': str
        }
    
    Returns:
        HTML 문자열
    """
    if not macro_indicators:
        return ""
    
    yield_slope = macro_indicators.get('yield_curve_slope')
    yield_status = macro_indicators.get('yield_curve_status', 'UNKNOWN')
    credit_change = macro_indicators.get('credit_spread_change')
    ted_spread = macro_indicators.get('ted_spread')
    interpretation = macro_indicators.get('interpretation', '')
    
    # Yield Curve 상태 색상
    if yield_status == 'INVERTED':
        yield_color = '#ef4444'  # 빨간색
        yield_icon = '⚠️'
    elif yield_status == 'FLAT':
        yield_color = '#eab308'  # 노란색
        yield_icon = '📊'
    else:
        yield_color = '#22c55e'  # 녹색
        yield_icon = '✅'
    
    # Yield Curve 시각화 (정상 vs 역전)
    yield_html = ""
    if yield_slope is not None:
        # Slope를 -200bp ~ +300bp 범위로 정규화하여 시각화
        normalized_slope = max(-200, min(300, yield_slope))
        bar_position = ((normalized_slope + 200) / 500) * 100  # 0-100% 범위
        
        # 0bp 기준선 표시
        zero_position = (200 / 500) * 100  # 0bp는 40% 위치
        
        yield_html = f"""
        <div class="yield-curve-visualization">
            <div class="yield-curve-bar-container">
                <div class="yield-curve-bar" style="width: 100%; height: 30px; background: linear-gradient(to right, #ef4444 0%, #ef4444 {zero_position}%, #22c55e {zero_position}%, #22c55e 100%); border-radius: 15px; position: relative;">
                    <div class="yield-curve-marker" style="position: absolute; left: {bar_position}%; top: 50%; transform: translate(-50%, -50%); width: 4px; height: 40px; background: white; border: 2px solid {yield_color}; border-radius: 2px; box-shadow: 0 0 8px {yield_color};"></div>
                    <div class="yield-curve-zero-line" style="position: absolute; left: {zero_position}%; top: 0; width: 2px; height: 100%; background: rgba(255,255,255,0.5);"></div>
                </div>
                <div class="yield-curve-labels">
                    <span style="color: #ef4444;">역전 (-)</span>
                    <span style="margin-left: auto; color: #22c55e;">정상 (+)</span>
                </div>
            </div>
            <div class="yield-curve-value" style="text-align: center; margin-top: 8px; font-size: 1.2rem; font-weight: 700; color: {yield_color};">
                {yield_icon} {yield_slope:.1f}bp ({yield_status})
            </div>
        </div>
        """
    
    # Credit Spread 변화율 HTML
    credit_html = ""
    if credit_change is not None:
        credit_color = '#ef4444' if credit_change < -5 else '#f97316' if credit_change < -2 else '#22c55e' if credit_change > 5 else '#eab308'
        credit_icon = '⚠️' if credit_change < -5 else '📊' if credit_change < -2 else '✅' if credit_change > 5 else '📊'
        
        credit_html = f"""
        <div class="credit-spread-item">
            <div class="credit-spread-label">신용 스프레드 변화 (20일)</div>
            <div class="credit-spread-value" style="color: {credit_color}; font-size: 1.5rem; font-weight: 700;">
                {credit_icon} {credit_change:+.1f}%
            </div>
        </div>
        """
    
    # TED Spread HTML
    ted_html = ""
    if ted_spread is not None:
        ted_color = '#ef4444' if ted_spread > 100 else '#f97316' if ted_spread > 50 else '#22c55e'
        ted_icon = '⚠️' if ted_spread > 100 else '📊' if ted_spread > 50 else '✅'
        
        ted_html = f"""
        <div class="ted-spread-item">
            <div class="ted-spread-label">TED Spread</div>
            <div class="ted-spread-value" style="color: {ted_color}; font-size: 1.5rem; font-weight: 700;">
                {ted_icon} {ted_spread:.1f}bp
            </div>
        </div>
        """
    
    return f"""
    <div class="cp-card cp-macro-environment">
        <h3>🌍 거시경제 환경 (Macro Environment)</h3>
        {yield_html if yield_html else '<p style="color: #9ca3af;">Yield Curve 데이터 없음</p>'}
        <div class="macro-metrics-grid">
            {credit_html if credit_html else ''}
            {ted_html if ted_html else ''}
        </div>
        {f'''<div class="macro-interpretation" style="margin-top: 16px; padding: 12px; background: rgba(139, 92, 246, 0.1); border-left: 3px solid #a78bfa; border-radius: 0 6px 6px 0; font-size: 0.9rem; color: #c4b5fd; line-height: 1.6;">
            {interpretation}
        </div>''' if interpretation else ''}
    </div>
    """


def generate_critical_path_section(critical_path_data: Dict) -> str:
    """
    Critical Path Analysis 섹션 HTML 생성
    
    Args:
        critical_path_data: CriticalPathResult.to_dict() 결과
    
    Returns:
        HTML 문자열 (데이터가 없으면 빈 문자열)
    """
    if not critical_path_data:
        return ""
    
    # 데이터 추출
    total_risk = critical_path_data.get('total_risk_score', 0)
    risk_level = critical_path_data.get('risk_level', 'LOW')
    # current_regime 기본값 처리 (None 또는 빈 문자열 체크)
    current_regime = critical_path_data.get('current_regime') or 'TRANSITION'
    if not current_regime or current_regime == 'None':
        current_regime = 'TRANSITION'
    # regime_confidence 기본값 처리
    regime_confidence = critical_path_data.get('regime_confidence')
    if regime_confidence is None:
        regime_confidence = 50.0
    transition_prob = critical_path_data.get('transition_probability', 0)
    path_contributions = critical_path_data.get('path_contributions', {})
    path_distribution = critical_path_data.get('path_distribution', {})  # 100% 정규화된 구성비
    risk_appetite_result = critical_path_data.get('risk_appetite_result', {})
    regime_result = critical_path_data.get('regime_result', {})
    active_warnings = critical_path_data.get('active_warnings', [])
    crypto_result = critical_path_data.get('crypto_result', {})
    
    # 위험도 색상 결정 (수정된 기준)
    if total_risk < 25:
        risk_color = '#22c55e'  # 녹색 (LOW)
    elif total_risk < 50:
        risk_color = '#eab308'  # 노란색 (MEDIUM)
    elif total_risk < 75:
        risk_color = '#f97316'  # 주황색 (HIGH)
    else:
        risk_color = '#ef4444'  # 빨간색 (CRITICAL)
    
    # 레짐 아이콘 및 색상
    regime_config = {
        'BULL': {'icon': '📈', 'color': '#22c55e'},
        'BEAR': {'icon': '📉', 'color': '#ef4444'},
        'TRANSITION': {'icon': '🌊', 'color': '#eab308'},
        'CRISIS': {'icon': '🚨', 'color': '#ef4444'}
    }
    regime_info = regime_config.get(current_regime, {'icon': '❓', 'color': '#9ca3af'})
    
    # 경로별 기여도 HTML
    path_names = {
        'liquidity': '유동성/금리',
        'concentration': 'AI/빅테크 집중',
        'credit': '신용 스트레스',
        'volatility': '변동성/공포',
        'rotation': '섹터 로테이션',
        'crypto': '암호화폐'
    }
    
    path_bars_html = ""
    # path_distribution이 있으면 우선 사용 (100% 정규화된 구성비)
    # 없으면 path_contributions 사용 (절대값)
    display_data = path_distribution if path_distribution else path_contributions
    
    if display_data:
        sorted_paths = sorted(display_data.items(), key=lambda x: x[1], reverse=True)
        
        # path_distribution을 사용하는 경우: 이미 100% 정규화되어 있음
        # path_contributions를 사용하는 경우: 최대값 대비 비율로 표시
        if path_distribution:
            # path_distribution: 이미 퍼센트 구성비 (0-100%)
            max_value = 100.0  # 최대값은 100%
        else:
            # path_contributions: 절대값이므로 최대값 대비 비율 계산
            max_value = max(display_data.values()) if display_data.values() else 100
        
        for path, value in sorted_paths:
            path_name = path_names.get(path, path)
            
            if path_distribution:
                # path_distribution: value가 이미 퍼센트 구성비
                bar_width = value  # 0-100% 범위
                display_value = value  # 퍼센트로 표시
            else:
                # path_contributions: 절대값을 최대값 대비 비율로 변환
                bar_width = (value / max_value * 100) if max_value > 0 else 0
                display_value = value  # 절대값으로 표시
            
            # 최대 기여도 경로 강조
            is_max = value == max(display_data.values()) if display_data.values() else False
            bar_color = '#ef4444' if is_max else '#60a5fa'
            
            # path_distribution 사용 시 "%" 표시, path_contributions 사용 시 절대값 표시
            if path_distribution:
                value_display = f"{display_value:.1f}%"
            else:
                value_display = f"{display_value:.1f}"
            
            path_bars_html += f"""
            <div class="path-bar-item">
                <div class="path-bar-label">
                    <span>{path_name}</span>
                    <span class="path-bar-value">{value_display}</span>
                </div>
                <div class="path-bar-container">
                    <div class="path-bar" style="width: {bar_width}%; background: {bar_color};"></div>
                </div>
            </div>
            """
    else:
        path_bars_html = '<p style="color: #9ca3af;">경로별 기여도 데이터 없음</p>'
    
    # Risk Appetite vs Uncertainty 매트릭스
    ra_score = risk_appetite_result.get('risk_appetite_score', 50)
    unc_score = risk_appetite_result.get('uncertainty_score', 50)
    market_state = risk_appetite_result.get('market_state', 'MIXED')
    
    # 매트릭스 위치 계산 (0-100을 0-200px로 변환)
    matrix_x = ra_score * 2  # 0-100 → 0-200px
    matrix_y = 200 - (unc_score * 2)  # Y축은 위에서 아래로 (0-100 → 200-0px)
    
    # 활성 경고 HTML
    warnings_html = ""
    if active_warnings:
        for warning in active_warnings[:5]:
            warnings_html += f"""
            <div class="warning-card">
                <span class="warning-icon">⚠️</span>
                <span class="warning-text">{warning}</span>
            </div>
            """
    else:
        warnings_html = '<p style="color: #9ca3af;">활성 경고 없음</p>'
    
    # Crypto Sentiment HTML
    crypto_html = ""
    if crypto_result:
        sentiment_score = crypto_result.get('sentiment_score', 50)
        sentiment_level = crypto_result.get('sentiment_level', 'NEUTRAL')
        btc_correlation = crypto_result.get('btc_spy_correlation', 0)
        correlation_regime = crypto_result.get('correlation_regime', 'DECOUPLED')
        is_leading = crypto_result.get('is_leading_indicator', False)
        leading_signal = crypto_result.get('leading_signal')
        causality_analysis = crypto_result.get('causality_analysis', {})
        
        # 센티먼트 색상
        if sentiment_score < 20:
            sentiment_color = '#ef4444'  # EXTREME_FEAR
        elif sentiment_score < 40:
            sentiment_color = '#f97316'  # FEAR
        elif sentiment_score < 60:
            sentiment_color = '#eab308'  # NEUTRAL
        elif sentiment_score < 80:
            sentiment_color = '#22c55e'  # GREED
        else:
            sentiment_color = '#10b981'  # EXTREME_GREED
        
        # Granger Causality 인과관계 해석
        causality_html = ""
        if causality_analysis and causality_analysis.get('relationship') != 'NO_CAUSALITY':
            relationship = causality_analysis.get('relationship', 'NO_CAUSALITY')
            x_to_y_pvalue = causality_analysis.get('x_to_y_pvalue', 1.0)
            y_to_x_pvalue = causality_analysis.get('y_to_x_pvalue', 1.0)
            optimal_lag = causality_analysis.get('optimal_lag', 0)
            
            if relationship == "X_LEADS":
                causality_text = f"BTC → SPY (p={x_to_y_pvalue:.3f}, 시차 {optimal_lag}일)"
                causality_color = '#60a5fa'
            elif relationship == "Y_LEADS":
                causality_text = f"SPY → BTC (p={y_to_x_pvalue:.3f}, 시차 {optimal_lag}일)"
                causality_color = '#a78bfa'
            elif relationship == "BIDIRECTIONAL":
                causality_text = f"양방향 인과관계 (BTC→SPY: p={x_to_y_pvalue:.3f}, SPY→BTC: p={y_to_x_pvalue:.3f})"
                causality_color = '#f97316'
            else:
                causality_text = "인과관계 없음"
                causality_color = '#9ca3af'
            
            causality_html = f"""
            <div class="crypto-metric">
                <div class="crypto-metric-label">Granger Causality</div>
                <div class="crypto-metric-value" style="color: {causality_color}; font-size: 0.9rem;">{causality_text}</div>
                <div class="crypto-metric-level" style="font-size: 0.75rem; color: #9ca3af;">인과관계 검정</div>
            </div>
            """
        
        crypto_html = f"""
        <div class="crypto-sentiment-card">
            <h4>🪙 암호화폐 센티먼트</h4>
            <div class="crypto-metrics">
                <div class="crypto-metric">
                    <div class="crypto-metric-label">센티먼트 점수</div>
                    <div class="crypto-metric-value" style="color: {sentiment_color};">{sentiment_score:.1f}</div>
                    <div class="crypto-metric-level">{sentiment_level}</div>
                </div>
                <div class="crypto-metric">
                    <div class="crypto-metric-label">BTC-SPY 상관관계</div>
                    <div class="crypto-metric-value">{btc_correlation:.2f}</div>
                    <div class="crypto-metric-level">{correlation_regime}</div>
                </div>
                {causality_html if causality_html else ''}
            </div>
            {f'<div class="leading-indicator-badge">🚨 선행지표: {leading_signal}</div>' if is_leading and leading_signal else ''}
        </div>
        """
    
    return f"""
    <!-- Critical Path Analysis 섹션 -->
    <div class="critical-path-section">
        <h2 class="section-title">🎯 Critical Path Analysis</h2>
        
        <!-- 섹션 A: Risk Overview -->
        <div class="cp-grid">
            <div class="cp-card cp-risk-overview">
                <h3>전체 위험도</h3>
                <div class="risk-gauge-container">
                    <div class="risk-gauge" style="--risk-value: {total_risk}; --risk-color: {risk_color};">
                        <div class="gauge-value">{total_risk:.1f}%</div>
                        <div class="gauge-level" style="color: {risk_color};">{risk_level}</div>
                    </div>
                </div>
                <div class="cp-note" style="font-size: 0.85rem; color: #9ca3af; margin-top: 12px; text-align: center;">
                    6개 위험 경로의 가중평균 (레짐: {current_regime})
                </div>
            </div>
            
            <!-- 섹션 C: Regime Status -->
            <div class="cp-card cp-regime-status">
                <h3>레짐 상태</h3>
                <div class="regime-status-content">
                    <div class="regime-status-icon" style="color: {regime_info['color']};">{regime_info['icon']}</div>
                    <div class="regime-status-label" style="color: {regime_info['color']};">{current_regime}</div>
                    <div class="regime-confidence">
                        <div class="confidence-label">레짐 확신도</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {regime_confidence}%;"></div>
                        </div>
                        <div class="confidence-value">{regime_confidence:.1f}%</div>
                    </div>
                    <div class="transition-prob">
                        <div class="transition-label">전환 확률</div>
                        <div class="transition-value" style="color: {'#ef4444' if transition_prob >= 50 else '#eab308' if transition_prob >= 30 else '#22c55e'};">{transition_prob:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 섹션 B: Path Contributions -->
        <div class="cp-card cp-path-contributions">
            <h3>경로별 위험 기여도</h3>
            <div class="path-bars">
                {path_bars_html}
            </div>
        </div>
        
        <!-- 섹션 D: Risk Appetite vs Uncertainty -->
        <div class="cp-card cp-risk-matrix">
            <h3>리스크 선호도 vs 불확실성</h3>
            <div class="matrix-container">
                <div class="matrix-grid">
                    <div class="matrix-quadrant" style="grid-area: 1 / 1 / 2 / 2; border-right: 2px solid rgba(255,255,255,0.2); border-bottom: 2px solid rgba(255,255,255,0.2);">
                        <div class="quadrant-label">CRISIS</div>
                    </div>
                    <div class="matrix-quadrant" style="grid-area: 1 / 2 / 2 / 3; border-bottom: 2px solid rgba(255,255,255,0.2);">
                        <div class="quadrant-label">SPECULATIVE</div>
                    </div>
                    <div class="matrix-quadrant" style="grid-area: 2 / 1 / 3 / 2; border-right: 2px solid rgba(255,255,255,0.2);">
                        <div class="quadrant-label">STAGNANT</div>
                    </div>
                    <div class="matrix-quadrant" style="grid-area: 2 / 2 / 3 / 3;">
                        <div class="quadrant-label">NORMAL</div>
                    </div>
                    <div class="matrix-marker" style="left: {matrix_x}px; top: {matrix_y}px;"></div>
                    <div class="matrix-marker-label" style="left: {matrix_x + 10}px; top: {matrix_y - 10}px;">
                        {market_state}
                    </div>
                </div>
                <div class="matrix-axes">
                    <div class="axis-label axis-y">Uncertainty (0-100)</div>
                    <div class="axis-label axis-x">Risk Appetite (0-100)</div>
                </div>
            </div>
            <div class="matrix-info">
                <div class="matrix-info-item">
                    <span>Risk Appetite: {ra_score:.1f}</span>
                </div>
                <div class="matrix-info-item">
                    <span>Uncertainty: {unc_score:.1f}</span>
                </div>
                <div class="matrix-info-item">
                    <span>Market State: {market_state}</span>
                </div>
            </div>
        </div>
        
        <!-- 섹션 E: Active Warnings -->
        {f'''<div class="cp-card cp-warnings">
            <h3>⚠️ 활성 경고 ({len(active_warnings)}개)</h3>
            <div class="warnings-list">
                {warnings_html}
            </div>
        </div>''' if active_warnings else ''}
        
        <!-- 섹션 F: Crypto Sentiment -->
        {crypto_html if crypto_html else ''}
        
        <!-- Spillover Analysis 상세 -->
        {_generate_spillover_section(critical_path_data.get('spillover_result', {})) if critical_path_data.get('spillover_result') else ''}
        
    </div>
    """


def _generate_llm_summary_section(llm_summary: str) -> str:
    """
    LLM 요약 섹션 HTML 생성
    
    Args:
        llm_summary: Claude API로 생성된 마크다운 요약
    
    Returns:
        HTML 문자열
    """
    if not llm_summary:
        return ""
    
    # 마크다운을 HTML로 변환
    html_content = llm_summary
    
    # 헤더 변환 (## → h3, ### → h4)
    html_content = re.sub(r'^## (.+)$', r'<h3 class="llm-h3">\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^### (.+)$', r'<h4 class="llm-h4">\1</h4>', html_content, flags=re.MULTILINE)
    
    # 볼드 변환 (**text** → <strong>text</strong>)
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    
    # 리스트 항목 변환 (- item → <li>item</li>)
    html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    
    # 숫자 리스트 변환 (1. item → <li>item</li>)
    html_content = re.sub(r'^\d+\. (.+)$', r'<li class="numbered">\1</li>', html_content, flags=re.MULTILINE)
    
    # 연속된 <li> 태그를 <ul>로 감싸기
    html_content = re.sub(r'((?:<li[^>]*>.*?</li>\s*)+)', r'<ul>\1</ul>', html_content, flags=re.DOTALL)
    
    # 줄바꿈 처리 (단, 이미 HTML 태그 앞뒤가 아닌 경우에만)
    html_content = re.sub(r'(?<!</h[34]>)\n(?!<)', '<br>\n', html_content)
    
    # 구분선 처리
    html_content = html_content.replace('---', '<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0;">')
    
    return f"""
    <div class="llm-summary-section" style="
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        ">
            <span style="font-size: 1.5em;">🧠</span>
            <h3 style="
                color: #f1f5f9;
                margin: 0;
                font-size: 1.25rem;
            ">AI 분석 요약</h3>
            <span style="
                font-size: 0.7em;
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3));
                padding: 3px 10px;
                border-radius: 12px;
                color: #93c5fd;
                font-weight: 600;
            ">Claude</span>
        </div>
        <div class="llm-summary-content" style="
            color: #cbd5e1;
            line-height: 1.75;
            font-size: 0.95rem;
        ">
            <style>
                .llm-summary-content .llm-h3 {{
                    color: #60a5fa;
                    font-size: 1.1rem;
                    margin: 1.25rem 0 0.75rem 0;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid rgba(96, 165, 250, 0.2);
                }}
                .llm-summary-content .llm-h4 {{
                    color: #a5b4fc;
                    font-size: 1rem;
                    margin: 1rem 0 0.5rem 0;
                }}
                .llm-summary-content ul {{
                    margin: 0.5rem 0;
                    padding-left: 1.5rem;
                    list-style: none;
                }}
                .llm-summary-content li {{
                    margin: 0.4rem 0;
                    padding-left: 0.5rem;
                    position: relative;
                }}
                .llm-summary-content li::before {{
                    content: "•";
                    color: #60a5fa;
                    font-weight: bold;
                    position: absolute;
                    left: -1rem;
                }}
                .llm-summary-content li.numbered::before {{
                    content: "";
                }}
                .llm-summary-content strong {{
                    color: #f1f5f9;
                }}
            </style>
            {html_content}
        </div>
        <div style="
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 0.75rem;
            color: #6b7280;
            text-align: right;
        ">
            Powered by Claude claude-sonnet-4-20250514 • 자동 생성된 분석입니다
        </div>
    </div>
    """


def generate_dashboard(
    signals: List[Dict],
    summary: str,
    interpretations: List[Dict],
    news: List[Dict],
    timestamp: str = None,
    regime_data: Dict = None,
    crypto_panel: Dict = None,
    crypto_collection_status: Dict = None,
    risk_data: Dict = None,  # NEW: ML 기반 위험 확률 데이터
    critical_path_data: Dict = None,  # NEW: Critical Path Analysis 결과
    signal_news: List[Dict] = None,  # NEW: Signal별 뉴스 정보
    risk_metrics: Dict[str, Dict] = None,  # NEW: 위험조정수익률 지표
    macro_indicators: Dict = None,  # NEW: 거시경제 선행지표
    llm_summary: str = None  # NEW: Claude API 기반 AI 요약
) -> str:
    """
    대시보드 HTML 생성
    
    v2.1 업데이트:
    - risk_data: ML 모델 기반 위험 확률 정보
      - enabled: Risk Model 활성화 여부
      - results: [{ticker, risk_prob, risk_level, model_type}, ...]
      - summary: 위험 요약 텍스트
    """
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if regime_data is None:
        regime_data = {}
    
    if crypto_panel is None:
        crypto_panel = {}
    
    if crypto_collection_status is None:
        crypto_collection_status = {}
    
    if critical_path_data is None:
        critical_path_data = {}
    
    if risk_data is None:
        risk_data = {'enabled': False, 'results': [], 'summary': ''}
    
    if signal_news is None:
        signal_news = []
    
    if risk_metrics is None:
        risk_metrics = {}
    
    if macro_indicators is None:
        macro_indicators = {}
    
    if llm_summary is None:
        llm_summary = ""
    
    # LLM 요약 섹션 생성
    llm_summary_html = _generate_llm_summary_section(llm_summary) if llm_summary else ""
    
    # ============================================================
    # ML Risk 통계 계산 (NEW in v2.1)
    # ============================================================
    risk_results = risk_data.get('results', [])
    risk_enabled = risk_data.get('enabled', False)
    
    # 위험 수준별 카운트
    risk_critical_count = len([r for r in risk_results if r.get('risk_level') == 'CRITICAL'])
    risk_high_count = len([r for r in risk_results if r.get('risk_level') == 'HIGH'])
    risk_medium_count = len([r for r in risk_results if r.get('risk_level') == 'MEDIUM'])
    risk_low_count = len([r for r in risk_results if r.get('risk_level') == 'LOW'])
    
    # 평균 위험 확률
    risk_probs = [r.get('risk_prob', 0) for r in risk_results if r.get('risk_prob') is not None]
    avg_risk_prob = sum(risk_probs) / len(risk_probs) * 100 if risk_probs else 0
    
    # 위험 레벨 딕셔너리 (ticker → risk info)
    risk_by_ticker = {r['ticker']: r for r in risk_results}
    
    # 신호 통계
    critical_count = len([s for s in signals if s.get('level') == 'CRITICAL'])
    alert_count = len([s for s in signals if s.get('level') == 'ALERT'])
    warning_count = len([s for s in signals if s.get('level') == 'WARNING'])
    total_count = len(signals)
    
    # 전체 상태 결정
    if critical_count > 0:
        overall_status = "CRITICAL"
        status_color = "#ef4444"
        status_text = "긴급 주의"
    elif alert_count > 0:
        overall_status = "ALERT"
        status_color = "#f97316"
        status_text = "주의 필요"
    elif warning_count > 0:
        overall_status = "WARNING"
        status_color = "#eab308"
        status_text = "관찰 권고"
    else:
        overall_status = "NORMAL"
        status_color = "#22c55e"
        status_text = "시장 안정"
    
    # 가장 활발한 자산 찾기
    ticker_counts = {}
    for s in signals:
        ticker = s.get('ticker', 'Unknown')
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    
    most_active = max(ticker_counts.items(), key=lambda x: x[1])[0] if ticker_counts else "N/A"
    
    # 신호 테이블 HTML 생성
    signal_rows = ""
    for s in signals[:30]:  # 최대 30개
        level = s.get('level', 'WARNING')
        level_color = {
            'CRITICAL': '#ef4444',
            'ALERT': '#f97316',
            'WARNING': '#eab308'
        }.get(level, '#22c55e')
        
        level_bg = {
            'CRITICAL': 'rgba(239, 68, 68, 0.1)',
            'ALERT': 'rgba(249, 115, 22, 0.1)',
            'WARNING': 'rgba(234, 179, 8, 0.1)'
        }.get(level, 'transparent')
        
        z_score = s.get('z_score', 0)
        z_display = f"{z_score:.2f}" if z_score != 0 else "-"
        
        time_str = s.get('timestamp', '')
        if time_str:
            try:
                time_str = datetime.fromisoformat(time_str).strftime("%H:%M:%S")
            except:
                time_str = time_str[-8:] if len(time_str) > 8 else time_str
        
        action_guide = s.get('action_guide', '')
        theory_note = s.get('theory_note', '')
        
        # Description with theory_note (for cross-asset anomalies)
        description_html = s.get('description', '')
        if theory_note:
            description_html += f'<div class="theory-note" style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px; font-style: italic;">📚 {theory_note[:150]}{"..." if len(theory_note) > 150 else ""}</div>'
        
        # ============================================================
        # ML Risk Probability (NEW in v2.1)
        # ============================================================
        ticker = s.get('ticker', '')
        risk_prob = s.get('risk_prob')  # main.py에서 병합된 값
        
        # 신호에 없으면 risk_by_ticker에서 조회
        if risk_prob is None and ticker in risk_by_ticker:
            risk_prob = risk_by_ticker[ticker].get('risk_prob')
        
        # Risk 표시 HTML 생성
        if risk_prob is not None:
            risk_pct = risk_prob * 100
            # 색상 결정 (LOW=녹색, MEDIUM=노랑, HIGH=주황, CRITICAL=빨강)
            if risk_pct >= 70:
                risk_color = '#ef4444'  # CRITICAL - 빨강
                risk_bg = 'rgba(239, 68, 68, 0.2)'
            elif risk_pct >= 50:
                risk_color = '#f97316'  # HIGH - 주황
                risk_bg = 'rgba(249, 115, 22, 0.2)'
            elif risk_pct >= 30:
                risk_color = '#eab308'  # MEDIUM - 노랑
                risk_bg = 'rgba(234, 179, 8, 0.2)'
            else:
                risk_color = '#22c55e'  # LOW - 녹색
                risk_bg = 'rgba(34, 197, 94, 0.2)'
            
            risk_html = f'<span class="risk-badge" style="background: {risk_bg}; color: {risk_color}; padding: 4px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem;">{risk_pct:.0f}%</span>'
        else:
            risk_html = '<span style="color: #6b7280;">-</span>'
        
        signal_rows += f"""
        <tr style="background: {level_bg};">
            <td><strong>{s.get('name', s.get('ticker', ''))}</strong></td>
            <td>{s.get('indicator', '')}</td>
            <td><span class="level-badge" style="background: {level_color};">{level}</span></td>
            <td>{risk_html}</td>
            <td>{z_display}</td>
            <td>{description_html}</td>
            <td class="action-guide">{action_guide}</td>
            <td>{time_str}</td>
        </tr>
        """
    
    # 해석 섹션 HTML
    interpretation_html = ""
    if interpretations:
        for interp in interpretations:
            text = interp.get('text', '')
            # 마크다운 볼드를 HTML로 변환
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = text.replace('\n', '<br>')
            interpretation_html += f"<div class='interpretation-item'>{text}</div>"
    
    # 뉴스 섹션 HTML
    news_html = ""
    if news:
        for n in news[:3]:  # 최대 3개
            signal_info = n.get('signal', {})
            news_text = n.get('news', '')
            # 마크다운 헤더와 볼드 변환
            news_text = re.sub(r'^##\s+(.+)$', r'<h4>\1</h4>', news_text, flags=re.MULTILINE)
            news_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', news_text)
            news_text = news_text.replace('\n', '<br>')
            
            news_html += f"""
            <div class="news-item">
                <div class="news-header">
                    <span class="news-ticker">{signal_info.get('ticker', '')}</span>
                    <span class="news-level" style="background: {{'CRITICAL': '#ef4444', 'ALERT': '#f97316', 'WARNING': '#eab308'}}.get(signal_info.get('level', ''), '#666');">
                        {signal_info.get('level', '')}
                    </span>
                </div>
                <div class="news-content">{news_text[:1500]}{'...' if len(news_text) > 1500 else ''}</div>
            </div>
            """
    
    # 요약 변환
    summary_html = summary.replace('## ', '<h3>').replace('### ', '<h4>')
    summary_html = summary_html.replace('**', '<strong>').replace('**', '</strong>')
    summary_html = summary_html.replace('\n', '<br>')
    
    # ============================================================
    # Regime 섹션 HTML 생성
    # ============================================================
    regime_summary = regime_data.get('summary', {})
    regimes = regime_data.get('regimes', {})
    sectors = regime_data.get('sectors', {})
    portfolio_rec = regime_data.get('portfolio_recommendation', {})
    
    global_regime = regime_summary.get('global_regime', 'N/A')
    bull_pct = regime_summary.get('bull_pct', 0)
    bear_pct = regime_summary.get('bear_pct', 0)
    bull_count = regime_summary.get('bull_count', 0)
    bear_count = regime_summary.get('bear_count', 0)
    
    # Regime 색상 및 아이콘
    regime_color = {'BULL': '#22c55e', 'BEAR': '#ef4444', 'MIXED': '#eab308', 'TRANSITION': '#eab308', 'CRISIS': '#ef4444'}.get(global_regime, '#6b7280')
    regime_icon = {'BULL': '🟢', 'BEAR': '🔴', 'MIXED': '🟡', 'TRANSITION': '🌊', 'CRISIS': '🚨'}.get(global_regime, '⚪')
    
    # Critical Path에서 실제 regime 정보 가져오기 (더 정확한 정보)
    if critical_path_data:
        actual_regime = critical_path_data.get('current_regime', global_regime)
        regime_confidence = critical_path_data.get('regime_confidence', 50.0)
        transition_prob = critical_path_data.get('transition_probability', 0.0)
    else:
        actual_regime = global_regime
        regime_confidence = 50.0
        transition_prob = 0.0
    
    # 자산군별 위험 현황 HTML 생성
    asset_class_config = {
        'equity': {'name': '주식', 'icon': '🏢'},
        'bond': {'name': '채권', 'icon': '📊'},
        'commodity': {'name': '원자재', 'icon': '🛢️'},
        'crypto': {'name': '암호화폐', 'icon': '🪙'},
        'reit': {'name': '리츠', 'icon': '🏠'},
        'fx': {'name': '환율', 'icon': '💱'},
        'pooled': {'name': '통합', 'icon': '🌐'},
        'unknown': {'name': '기타', 'icon': '❓'}
    }
    
    # 자산군별 통계 계산
    asset_class_stats = {}
    for result in risk_results:
        model_type = result.get('model_type', 'unknown')
        
        # model_type에서 자산군 추출
        if 'logistic_' in model_type:
            asset_class = model_type.replace('logistic_', '').replace('_pooled', 'pooled')
        elif 'heuristic' in model_type:
            asset_class = 'fx'
        elif 'pooled' in model_type:
            asset_class = 'pooled'
        else:
            asset_class = 'unknown'
        
        if asset_class not in asset_class_stats:
            asset_class_stats[asset_class] = {'probs': [], 'high_risk': 0}
        
        risk_prob = result.get('risk_prob', 0)
        if risk_prob is not None:
            asset_class_stats[asset_class]['probs'].append(risk_prob)
        
        if result.get('risk_level') in ['HIGH', 'CRITICAL']:
            asset_class_stats[asset_class]['high_risk'] += 1
    
    # 자산군별 HTML 생성
    asset_class_html = ""
    # 표시할 자산군 순서
    display_order = ['equity', 'bond', 'commodity', 'crypto', 'reit', 'fx', 'pooled', 'unknown']
    
    for asset_class in display_order:
        if asset_class not in asset_class_stats:
            continue
        
        stats = asset_class_stats[asset_class]
        config = asset_class_config.get(asset_class, {'name': asset_class, 'icon': '❓'})
        
        # 평균 위험 확률 계산
        if stats['probs']:
            avg_prob = sum(stats['probs']) / len(stats['probs']) * 100
        else:
            avg_prob = 0.0
        
        # 상태 결정
        if avg_prob < 25:
            status = "안정"
            status_color = '#22c55e'  # 녹색
        elif avg_prob < 40:
            status = "주의"
            status_color = '#eab308'  # 노란색
        elif avg_prob < 55:
            status = "경고"
            status_color = '#f97316'  # 주황색
        else:
            status = "위험"
            status_color = '#ef4444'  # 빨간색
        
        high_risk_count = stats['high_risk']
        total_count = len(stats['probs'])
        
        asset_class_html += f"""
        <div class="asset-class-item" style="border-left: 3px solid {status_color};">
            <div class="asset-class-header">
                <span class="asset-class-icon">{config['icon']}</span>
                <span class="asset-class-name">{config['name']}</span>
                <span class="asset-class-status" style="color: {status_color};">{status}</span>
            </div>
            <div class="asset-class-body">
                <div class="asset-class-bar-container">
                    <div class="asset-class-bar" style="width: {min(avg_prob, 100)}%; background: {status_color};"></div>
                </div>
                <div class="asset-class-stats">
                    <span class="asset-class-avg">평균: {avg_prob:.1f}%</span>
                    <span class="asset-class-high-risk">고위험: {high_risk_count}/{total_count}</span>
                </div>
            </div>
        </div>
        """
    
    if not asset_class_html:
        asset_class_html = '<p style="color: #9ca3af;">자산군별 위험 데이터 없음</p>'
    
    # 포트폴리오 추천 HTML
    asset_allocation = portfolio_rec.get('asset_allocation', {})
    allocation_html = ""
    for asset, pct in asset_allocation.items():
        bar_color = {
            'equity': '#3b82f6', 
            'bond': '#22c55e', 
            'gold': '#eab308', 
            'crypto': '#f97316',  # Crypto 추가
            'cash': '#9ca3af'
        }.get(asset, '#6b7280')
        asset_name = {
            'equity': '주식', 
            'bond': '채권', 
            'gold': '금', 
            'crypto': '암호화폐',  # Crypto 추가
            'cash': '현금'
        }.get(asset, asset)
        allocation_html += f"""
        <div class="allocation-item">
            <span class="allocation-name">{asset_name}</span>
            <div class="allocation-bar">
                <div class="allocation-fill" style="width: {pct}%; background: {bar_color};"></div>
            </div>
            <span class="allocation-pct">{pct}%</span>
        </div>
        """
    
    # Crypto 추천 메모
    crypto_note = portfolio_rec.get('crypto_note', '')
    
    # Overweight/Underweight 섹터
    overweight = portfolio_rec.get('sector_overweight', [])
    underweight = portfolio_rec.get('sector_underweight', [])
    
    overweight_html = ", ".join([f"{s.get('ticker', '')} ({s.get('sector', '')})" for s in overweight[:5]]) or "없음"
    underweight_html = ", ".join([f"{s.get('ticker', '')} ({s.get('sector', '')})" for s in underweight[:5]]) or "없음"
    
    # Crypto 패널 HTML 생성
    crypto_signals = crypto_panel.get('signals', [])
    crypto_news_list = crypto_panel.get('news', [])
    crypto_panel_html = ""
    if crypto_panel and (crypto_signals or crypto_news_list or crypto_collection_status):
        crypto_panel_html = generate_crypto_panel_html(
            crypto_signals=crypto_signals,
            crypto_news=crypto_news_list,
            crypto_collection_status=crypto_collection_status
        )
    
    # MA Status 섹션 생성
    regime_result = regime_data.get('regime_result', {})
    ma_status = regime_result.get('ma_status', {}) if isinstance(regime_result, dict) else {}
    ma_status_html = _generate_ma_status_section(ma_status) if ma_status else ""
    
    # Signal News 섹션 생성
    # signal_news가 None이거나 빈 리스트인 경우 빈 문자열 반환
    signal_news_html = _generate_signal_news_section(signal_news) if signal_news and len(signal_news) > 0 else ""
    
    # Risk Summary 섹션 생성
    risk_summary_text = risk_data.get('summary', '')
    risk_summary_html = _generate_risk_summary_section(risk_summary_text) if risk_summary_text and risk_summary_text != "위험 모델 미적용" else ""
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="3600">
    <title>Market Anomaly Dashboard - {timestamp[:10]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js" 
            onerror="window.chartJsLoadFailed = true; console.error('Chart.js CDN 로드 실패');"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        /* 헤더 */
        .header {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .header-left h1 {{
            font-size: 1.8rem;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header-left .timestamp {{
            color: #9ca3af;
            font-size: 0.95rem;
        }}
        
        .header-right {{
            display: flex;
            align-items: center;
            gap: 24px;
        }}
        
        .status-badge {{
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            color: white;
            background: {status_color};
        }}
        
        .signal-count {{
            text-align: center;
        }}
        
        .signal-count .number {{
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
        }}
        
        .signal-count .label {{
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        /* 그리드 레이아웃 */
        .grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .card h3 {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .card .value {{
            font-size: 2.2rem;
            font-weight: 700;
        }}
        
        .card .sub {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: 4px;
        }}
        
        .critical {{ color: #ef4444; }}
        .alert {{ color: #f97316; }}
        .warning {{ color: #eab308; }}
        .normal {{ color: #22c55e; }}
        
        /* 차트 섹션 */
        .chart-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        .chart-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .chart-card h3 {{
            font-size: 1.1rem;
            margin-bottom: 16px;
            color: #e4e4e7;
        }}
        
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        
        /* 테이블 */
        .table-section {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-x: auto;
        }}
        
        .table-section h3 {{
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: #e4e4e7;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
            color: #9ca3af;
            position: sticky;
            top: 0;
        }}
        
        .level-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }}
        
        .action-guide {{
            max-width: 300px;
            font-size: 0.85rem;
            color: #93c5fd;
        }}
        
        /* AI 요약 */
        .summary-section {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}
        
        .summary-section h3 {{
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: #a78bfa;
        }}
        
        .summary-content {{
            line-height: 1.8;
        }}
        
        .summary-content h3, .summary-content h4 {{
            color: #c4b5fd;
            margin: 16px 0 8px 0;
        }}
        
        /* 해석 섹션 */
        .interpretation-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .interpretation-item {{
            padding: 16px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            margin-bottom: 12px;
            line-height: 1.7;
        }}
        
        /* 뉴스 섹션 */
        .news-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .news-item {{
            padding: 16px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            margin-bottom: 16px;
        }}
        
        .news-header {{
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }}
        
        .news-ticker {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .news-level {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            color: white;
        }}
        
        .news-content {{
            font-size: 0.9rem;
            line-height: 1.6;
            color: #d1d5db;
        }}
        
        .news-content h4 {{
            color: #93c5fd;
            margin: 12px 0 8px 0;
        }}
        
        /* Regime 섹션 스타일 */
        .regime-section {{
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        .regime-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .regime-card h3 {{
            margin-bottom: 16px;
            font-size: 1.1rem;
            color: #a5b4fc;
        }}
        
        .regime-main {{
            text-align: center;
            padding: 20px;
        }}
        
        .regime-icon {{
            font-size: 3rem;
            margin-bottom: 8px;
        }}
        
        .regime-label {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        
        .regime-stats {{
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 16px;
        }}
        
        .regime-stat {{
            text-align: center;
        }}
        
        .regime-stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .regime-stat-label {{
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        .sector-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
            margin-bottom: 6px;
        }}
        
        .sector-icon {{
            font-size: 0.9rem;
        }}
        
        .sector-name {{
            flex: 1;
            font-size: 0.9rem;
        }}
        
        .sector-ticker {{
            color: #60a5fa;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .sector-cross {{
            color: #9ca3af;
            font-size: 0.75rem;
        }}
        
        .sector-conf {{
            color: #a5b4fc;
            font-size: 0.8rem;
        }}
        
        /* 자산군별 위험 현황 스타일 */
        .asset-class-item {{
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            margin-bottom: 12px;
        }}
        
        .asset-class-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .asset-class-icon {{
            font-size: 1.2rem;
        }}
        
        .asset-class-name {{
            flex: 1;
            font-size: 1rem;
            font-weight: 600;
            color: #e4e4e7;
        }}
        
        .asset-class-status {{
            font-size: 0.85rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .asset-class-body {{
            margin-top: 8px;
        }}
        
        .asset-class-bar-container {{
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        
        .asset-class-bar {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }}
        
        .asset-class-stats {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        .asset-class-avg {{
            font-weight: 600;
            color: #d1d5db;
        }}
        
        .asset-class-high-risk {{
            color: #fca5a5;
        }}
        
        /* Regime Confidence 스타일 */
        .regime-confidence-section {{
            margin-top: 16px;
        }}
        
        .regime-confidence-item {{
            margin-bottom: 12px;
        }}
        
        .regime-confidence-label {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 6px;
        }}
        
        .regime-confidence-bar-container {{
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 4px;
        }}
        
        .regime-confidence-bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        
        .regime-confidence-value {{
            font-size: 0.9rem;
            color: #d1d5db;
            text-align: right;
        }}
        
        .regime-transition-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .regime-transition-label {{
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        .regime-transition-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .allocation-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        
        .allocation-name {{
            width: 50px;
            font-size: 0.9rem;
        }}
        
        .allocation-bar {{
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .allocation-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .allocation-pct {{
            width: 40px;
            text-align: right;
            font-weight: 600;
        }}
        
        .sector-rec {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .sector-rec-label {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 4px;
        }}
        
        .sector-rec-value {{
            font-size: 0.9rem;
            color: #e4e4e7;
        }}
        
        /* Crypto Panel 스타일 */
        .crypto-panel {{
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(234, 179, 8, 0.1) 100%);
            border: 1px solid rgba(249, 115, 22, 0.3);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .crypto-panel h3 {{
            color: #f97316;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }}
        
        .crypto-panel h4 {{
            color: #fbbf24;
            margin: 16px 0 12px 0;
            font-size: 1rem;
        }}
        
        .crypto-collection-status {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        
        .crypto-status {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1rem;
        }}
        
        .crypto-status .status-icon {{
            font-size: 1.2rem;
        }}
        
        .crypto-status .fallback-badge {{
            background: #eab308;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .crypto-status-details {{
            margin-top: 12px;
            display: grid;
            gap: 6px;
        }}
        
        .crypto-status-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        
        .crypto-status-item .ticker {{
            font-weight: 600;
            color: #f97316;
            width: 80px;
        }}
        
        .crypto-status-item .name {{
            flex: 1;
            color: #e4e4e7;
        }}
        
        .crypto-status-item .source {{
            color: #9ca3af;
            font-size: 0.75rem;
        }}
        
        .crypto-signals {{
            margin-bottom: 16px;
        }}
        
        .crypto-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .crypto-table th,
        .crypto-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .crypto-table th {{
            background: rgba(0, 0, 0, 0.2);
            color: #f97316;
            font-weight: 600;
        }}
        
        .crypto-news {{
            margin-top: 16px;
        }}
        
        .crypto-news-item {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-left: 3px solid #f97316;
        }}
        
        .crypto-news-item .news-ticker {{
            color: #f97316;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }}
        
        .crypto-news-item .news-headline {{
            color: #e4e4e7;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }}
        
        .crypto-news-item .news-summary {{
            color: #9ca3af;
            font-size: 0.85rem;
            line-height: 1.4;
        }}
        
        .crypto-note {{
            background: rgba(249, 115, 22, 0.1);
            border-left: 3px solid #f97316;
            padding: 10px 14px;
            margin-top: 12px;
            border-radius: 0 6px 6px 0;
            font-size: 0.85rem;
            color: #fbbf24;
        }}
        
        /* Risk Panel 스타일 (NEW in v2.1) */
        .risk-panel {{
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .risk-panel h3 {{
            color: #a78bfa;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }}
        
        .risk-panel h4 {{
            color: #c4b5fd;
            margin-bottom: 12px;
            font-size: 1rem;
        }}
        
        .risk-panel-grid {{
            display: grid;
            grid-template-columns: 1fr 1.5fr 1.5fr;
            gap: 20px;
            margin-bottom: 16px;
        }}
        
        .risk-card {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 16px;
        }}
        
        .risk-summary {{
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        
        .risk-summary-icon {{
            font-size: 2.5rem;
            margin-bottom: 8px;
        }}
        
        .risk-summary-value {{
            font-size: 2.2rem;
            font-weight: 700;
        }}
        
        .risk-summary-label {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-top: 4px;
        }}
        
        .risk-summary-sub {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 4px;
        }}
        
        .risk-level-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .risk-level-item:last-child {{
            border-bottom: none;
        }}
        
        .risk-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .risk-label {{
            flex: 1;
            font-size: 0.9rem;
        }}
        
        .risk-count {{
            font-weight: 600;
            font-size: 1.1rem;
            color: #e4e4e7;
        }}
        
        .risk-asset-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .risk-asset-item:last-child {{
            border-bottom: none;
        }}
        
        .risk-ticker {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .risk-prob {{
            font-weight: 600;
        }}
        
        .risk-note {{
            background: rgba(139, 92, 246, 0.1);
            border-left: 3px solid #a78bfa;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #c4b5fd;
            line-height: 1.5;
        }}
        
        .risk-note p {{
            margin: 4px 0;
        }}
        
        /* 전체 자산별 위험확률 테이블 (NEW) */
        .risk-table-section {{
            margin-top: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 16px;
        }}
        
        .risk-table-section h4 {{
            color: #60a5fa;
            margin-bottom: 12px;
            font-size: 1rem;
        }}
        
        .risk-table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .risk-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        
        .risk-table th,
        .risk-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        
        .risk-table th {{
            background: rgba(96, 165, 250, 0.15);
            color: #93c5fd;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .risk-table tbody tr:hover {{
            background: rgba(255, 255, 255, 0.05) !important;
        }}
        
        .risk-bar-container {{
            width: 80px;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .risk-bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        @media (max-width: 1200px) {{
            .risk-panel-grid {{ grid-template-columns: 1fr; }}
        }}
        
        /* Critical Path 섹션 스타일 */
        .section-title {{
            font-size: 1.8rem;
            margin: 32px 0 24px 0;
            color: #a5b4fc;
            border-bottom: 2px solid rgba(165, 180, 252, 0.3);
            padding-bottom: 12px;
        }}
        
        .critical-path-section {{
            margin-bottom: 32px;
        }}
        
        .cp-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        .cp-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .cp-card h3 {{
            margin-bottom: 20px;
            font-size: 1.2rem;
            color: #a5b4fc;
        }}
        
        /* Risk Gauge */
        .risk-gauge-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .risk-gauge {{
            position: relative;
            width: 200px;
            height: 100px;
        }}
        
        .risk-gauge::before {{
            content: '';
            position: absolute;
            width: 200px;
            height: 100px;
            border-radius: 200px 200px 0 0;
            background: conic-gradient(
                from 180deg at 50% 100%,
                var(--risk-color) 0deg,
                var(--risk-color) calc(var(--risk-value) * 1.8deg),
                rgba(255, 255, 255, 0.1) calc(var(--risk-value) * 1.8deg),
                rgba(255, 255, 255, 0.1) 180deg
            );
            mask: radial-gradient(circle at 50% 100%, transparent 70px, black 70px);
            -webkit-mask: radial-gradient(circle at 50% 100%, transparent 70px, black 70px);
        }}
        
        .risk-gauge::after {{
            content: '';
            position: absolute;
            width: 140px;
            height: 70px;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 140px 140px 0 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            z-index: 1;
        }}
        
        .gauge-value {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--risk-color);
            z-index: 2;
        }}
        
        .gauge-level {{
            position: absolute;
            top: 70%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 1rem;
            font-weight: 600;
            z-index: 2;
        }}
        
        /* Regime Status */
        .regime-status-content {{
            text-align: center;
        }}
        
        .regime-status-icon {{
            font-size: 3rem;
            margin-bottom: 8px;
        }}
        
        .regime-status-label {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        
        .regime-confidence {{
            margin-bottom: 16px;
        }}
        
        .confidence-label {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 8px;
        }}
        
        .confidence-bar {{
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 4px;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #22c55e, #eab308);
            transition: width 0.3s;
        }}
        
        .confidence-value {{
            font-size: 0.9rem;
            color: #d1d5db;
        }}
        
        .transition-prob {{
            margin-top: 12px;
        }}
        
        .transition-label {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 4px;
        }}
        
        .transition-value {{
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        /* Path Contributions */
        .path-bars {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        
        .path-bar-item {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .path-bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            color: #d1d5db;
        }}
        
        .path-bar-value {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .path-bar-container {{
            width: 100%;
            height: 24px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .path-bar {{
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s;
        }}
        
        /* Risk Matrix */
        .matrix-container {{
            position: relative;
            margin-bottom: 16px;
        }}
        
        .matrix-grid {{
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
        }}
        
        .matrix-quadrant {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
        }}
        
        .quadrant-label {{
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.5);
            text-align: center;
        }}
        
        .matrix-marker {{
            position: absolute;
            width: 12px;
            height: 12px;
            background: #ef4444;
            border: 2px solid white;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            z-index: 10;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
        }}
        
        .matrix-marker-label {{
            position: absolute;
            font-size: 0.75rem;
            color: #ef4444;
            font-weight: 600;
            background: rgba(0, 0, 0, 0.7);
            padding: 2px 6px;
            border-radius: 4px;
            z-index: 11;
        }}
        
        .matrix-axes {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
            padding: 0 20px;
            font-size: 0.75rem;
            color: #9ca3af;
        }}
        
        .axis-label {{
            font-size: 0.75rem;
        }}
        
        .axis-x {{
            flex: 1;
            text-align: center;
        }}
        
        .axis-y {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
        }}
        
        .matrix-info {{
            display: flex;
            justify-content: space-around;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .matrix-info-item {{
            font-size: 0.9rem;
            color: #d1d5db;
        }}
        
        /* Warnings */
        .warnings-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .warning-card {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            border-radius: 6px;
        }}
        
        .warning-icon {{
            font-size: 1.2rem;
        }}
        
        .warning-text {{
            flex: 1;
            color: #fca5a5;
            font-size: 0.95rem;
        }}
        
        /* Crypto Sentiment */
        .crypto-sentiment-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 24px;
        }}
        
        .crypto-sentiment-card h4 {{
            margin-bottom: 16px;
            font-size: 1.1rem;
            color: #a5b4fc;
        }}
        
        .crypto-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }}
        
        .crypto-metric {{
            text-align: center;
        }}
        
        .crypto-metric-label {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 8px;
        }}
        
        .crypto-metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .crypto-metric-level {{
            font-size: 0.85rem;
            color: #d1d5db;
        }}
        
        .leading-indicator-badge {{
            margin-top: 12px;
            padding: 8px 12px;
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            border-radius: 6px;
            color: #fca5a5;
            font-size: 0.9rem;
            text-align: center;
        }}
        
        @media (max-width: 1200px) {{
            .cp-grid {{ grid-template-columns: 1fr; }}
            .crypto-metrics {{ grid-template-columns: 1fr; }}
        }}
        
        /* Signal News 섹션 스타일 */
        .signal-news-section {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .signal-news-section h3 {{
            margin-bottom: 16px;
            font-size: 1.2rem;
            color: #a5b4fc;
        }}
        
        .signal-news-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .signal-news-item {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .signal-news-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        
        .signal-news-header:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .signal-news-info {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .signal-news-ticker {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .signal-news-name {{
            color: #e4e4e7;
        }}
        
        .signal-news-level {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            color: white;
            font-weight: 600;
        }}
        
        .signal-news-toggle {{
            color: #9ca3af;
            font-size: 0.9rem;
            transition: transform 0.2s;
        }}
        
        .signal-news-content {{
            padding: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .signal-news-description {{
            font-size: 0.9rem;
            color: #d1d5db;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .signal-news-text {{
            font-size: 0.85rem;
            line-height: 1.6;
            color: #9ca3af;
        }}
        
        /* Spillover 섹션 스타일 */
        .spillover-detail-section {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .spillover-detail-section h4 {{
            margin-bottom: 16px;
            font-size: 1.1rem;
            color: #a5b4fc;
        }}
        
        .spillover-detail-section h5 {{
            margin: 16px 0 12px 0;
            font-size: 0.95rem;
            color: #c4b5fd;
        }}
        
        .spillover-summary {{
            display: flex;
            gap: 24px;
            margin-bottom: 20px;
        }}
        
        .spillover-metric {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .spillover-metric-label {{
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        .spillover-metric-value {{
            font-size: 1.3rem;
            font-weight: 700;
        }}
        
        .spillover-paths {{
            margin-bottom: 20px;
        }}
        
        .spillover-path-item {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }}
        
        .spillover-path-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }}
        
        .spillover-source {{
            font-weight: 600;
            color: #ef4444;
        }}
        
        .spillover-arrow {{
            color: #9ca3af;
        }}
        
        .spillover-target {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .spillover-category {{
            margin-left: auto;
            font-size: 0.75rem;
            color: #9ca3af;
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        .spillover-path-strength {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .spillover-strength-bar {{
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .spillover-strength-value {{
            font-size: 0.85rem;
            color: #d1d5db;
            min-width: 40px;
        }}
        
        .spillover-impacts {{
            margin-top: 16px;
        }}
        
        .spillover-impact-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .impact-asset {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .impact-value {{
            font-weight: 600;
        }}
        
        /* MA Status 섹션 스타일 */
        .ma-status-section {{
            padding: 16px;
        }}
        
        .ma-status-section h4 {{
            margin-bottom: 16px;
            font-size: 1.1rem;
            color: #a5b4fc;
        }}
        
        .ma-values {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
        }}
        
        .ma-value-item {{
            flex: 1;
            text-align: center;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }}
        
        .ma-label {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 4px;
        }}
        
        .ma-value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #e4e4e7;
        }}
        
        .ma-slope {{
            font-size: 1.5rem;
            margin-top: 4px;
        }}
        
        .ma-deviations {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        
        .ma-deviation-item {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .ma-deviation-label {{
            font-size: 0.85rem;
            color: #9ca3af;
        }}
        
        .ma-deviation-bar-container {{
            position: relative;
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .ma-deviation-bar {{
            position: absolute;
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s;
        }}
        
        .ma-deviation-value {{
            font-size: 0.9rem;
            font-weight: 600;
            text-align: right;
        }}
        
        /* Risk Summary 섹션 스타일 */
        .risk-summary-section {{
            background: rgba(139, 92, 246, 0.1);
            border-left: 3px solid #a78bfa;
            border-radius: 0 8px 8px 0;
            padding: 16px;
            margin-bottom: 20px;
        }}
        
        .risk-summary-section h4 {{
            margin-bottom: 12px;
            font-size: 1rem;
            color: #a78bfa;
        }}
        
        .risk-summary-content {{
            font-size: 0.9rem;
            line-height: 1.6;
            color: #c4b5fd;
        }}
        
        /* Risk Metrics 섹션 스타일 */
        .risk-metrics-section {{
            margin-top: 24px;
            padding-top: 24px;
            border-top: 2px solid rgba(255, 255, 255, 0.1);
        }}
        
        .risk-metrics-section h4 {{
            margin-bottom: 12px;
            font-size: 1.1rem;
            color: #a78bfa;
        }}
        
        .risk-metrics-note {{
            margin-bottom: 16px;
            padding: 12px;
            background: rgba(139, 92, 246, 0.05);
            border-radius: 6px;
            border-left: 3px solid #a78bfa;
        }}
        
        .risk-metrics-table-container {{
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .risk-metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .risk-metrics-table th,
        .risk-metrics-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .risk-metrics-table th {{
            background: rgba(139, 92, 246, 0.15);
            color: #c4b5fd;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .risk-metrics-table tbody tr:hover {{
            background: rgba(255, 255, 255, 0.05) !important;
        }}
        
        /* Markov Switching Regime 섹션 스타일 */
        .markov-regime-section {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .markov-regime-section h3 {{
            color: #a78bfa;
            margin-bottom: 16px;
            font-size: 1.3rem;
        }}
        
        .markov-note-intro {{
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 20px;
            padding: 12px;
            background: rgba(139, 92, 246, 0.05);
            border-radius: 6px;
            border-left: 3px solid #a78bfa;
        }}
        
        .markov-regime-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .markov-regime-card h4 {{
            color: #c4b5fd;
            margin-bottom: 16px;
            font-size: 1.1rem;
        }}
        
        .markov-regime-card h5 {{
            color: #a5b4fc;
            margin: 16px 0 12px 0;
            font-size: 1rem;
        }}
        
        .markov-note {{
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 12px;
        }}
        
        .markov-transition-section {{
            margin-bottom: 20px;
        }}
        
        .transition-matrix-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .transition-matrix-table th,
        .transition-matrix-table td {{
            padding: 10px 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .transition-matrix-table th {{
            background: rgba(139, 92, 246, 0.2);
            color: #c4b5fd;
            font-weight: 600;
        }}
        
        .markov-metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }}
        
        .markov-metric-card {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 16px;
        }}
        
        .duration-list,
        .next-prob-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .duration-item,
        .next-prob-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .duration-item:last-child,
        .next-prob-item:last-child {{
            border-bottom: none;
        }}
        
        .duration-regime,
        .next-prob-regime {{
            color: #d1d5db;
            font-weight: 600;
        }}
        
        .duration-value,
        .next-prob-value {{
            color: #60a5fa;
            font-weight: 600;
        }}
        
        .markov-chart-section {{
            margin-top: 20px;
        }}
        
        @media (max-width: 1200px) {{
            .markov-metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* 반응형 */
        @media (max-width: 1200px) {{
            .grid {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-section {{ grid-template-columns: 1fr; }}
            .regime-section {{ grid-template-columns: 1fr; }}
        }}
        
        @media (max-width: 768px) {{
            .header {{ flex-direction: column; gap: 16px; text-align: center; }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <div class="header">
            <div class="header-left">
                <h1>📊 Market Anomaly Dashboard</h1>
                <div class="timestamp">업데이트: {timestamp} | 1시간 캐시</div>
            </div>
            <div class="header-right">
                <div class="signal-count">
                    <div class="number">{total_count}</div>
                    <div class="label">감지된 신호</div>
                </div>
                <div class="status-badge">{status_text}</div>
            </div>
        </div>
        
        <!-- 신호 카드 -->
        <div class="grid">
            <div class="card">
                <h3>🔴 Critical 신호</h3>
                <div class="value critical">{critical_count}</div>
                <div class="sub">즉시 주의 필요</div>
            </div>
            <div class="card">
                <h3>🟠 Alert 신호</h3>
                <div class="value alert">{alert_count}</div>
                <div class="sub">모니터링 강화</div>
            </div>
            <div class="card">
                <h3>🟡 Warning 신호</h3>
                <div class="value warning">{warning_count}</div>
                <div class="sub">추세 관찰</div>
            </div>
            <div class="card">
                <h3>🤖 ML 평균 위험</h3>
                <div class="value" style="font-size: 1.8rem; color: {'#ef4444' if avg_risk_prob >= 50 else '#eab308' if avg_risk_prob >= 30 else '#22c55e'};">{avg_risk_prob:.1f}%</div>
                <div class="sub">{'High ' + str(risk_high_count) + ' / Critical ' + str(risk_critical_count) if risk_enabled else 'Risk Model 미적용'}</div>
            </div>
            <div class="card">
                <h3>📈 가장 활발한 자산</h3>
                <div class="value" style="font-size: 1.5rem; color: #60a5fa;">{most_active}</div>
                <div class="sub">{ticker_counts.get(most_active, 0)}개 신호 동시 발생</div>
            </div>
        </div>
        
        <!-- 차트 섹션 -->
        <div class="chart-section">
            <div class="chart-card">
                <h3>📊 신호 분포 현황</h3>
                <div class="chart-container">
                    <canvas id="signalChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>📈 지표별 신호 분포</h3>
                <div class="chart-container">
                    <canvas id="indicatorChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Critical Path Analysis 섹션 -->
        {generate_critical_path_section(critical_path_data)}
        
        <!-- 거시경제 환경 섹션 (NEW) -->
        {_generate_macro_environment_section(macro_indicators) if macro_indicators else ''}
        
        <!-- Regime 분석 섹션 -->
        <div class="regime-section">
            <div class="regime-card regime-main">
                <h3>📈 시장 국면 (Regime)</h3>
                {generate_regime_display(critical_path_data if critical_path_data else {'current_regime': actual_regime, 'regime_confidence': regime_confidence, 'transition_probability': transition_prob})}
            </div>
            
            <div class="regime-card">
                {generate_asset_risk_section(signals)}
            </div>
            
            {f'''<div class="regime-card">
                {ma_status_html}
            </div>''' if ma_status_html else ''}
            
            <!-- Markov Switching 분석 -->
            {_generate_markov_regime_section(regime_data.get('markov_analysis', {})) if regime_data.get('markov_analysis') else ''}
            
            <div class="regime-card">
                <h3>💼 포트폴리오 추천 (Moderate)</h3>
                {allocation_html if allocation_html else '<p style="color: #9ca3af;">추천 데이터 없음</p>'}
                <div class="sector-rec">
                    <div class="sector-rec-label">📈 Overweight</div>
                    <div class="sector-rec-value" style="color: #22c55e;">{overweight_html}</div>
                </div>
                <div class="sector-rec" style="margin-top: 8px; padding-top: 8px; border-top: none;">
                    <div class="sector-rec-label">📉 Underweight</div>
                    <div class="sector-rec-value" style="color: #ef4444;">{underweight_html}</div>
                </div>
                {f'<div class="crypto-note">🪙 {crypto_note}</div>' if crypto_note else ''}
            </div>
        </div>
        
        <!-- Crypto 패널 -->
        {crypto_panel_html}
        
        <!-- Risk Model 패널 (NEW in v2.1) -->
        {f'''<div class="risk-panel">
            <h3>🤖 ML 기반 위험 분석 (Risk Model)</h3>
            {risk_summary_html if risk_summary_html else ''}
            <div class="risk-panel-grid">
                <div class="risk-card risk-summary">
                    <div class="risk-summary-icon">{'🔴' if avg_risk_prob >= 50 else '🟡' if avg_risk_prob >= 30 else '🟢'}</div>
                    <div class="risk-summary-value" style="color: {'#ef4444' if avg_risk_prob >= 50 else '#eab308' if avg_risk_prob >= 30 else '#22c55e'};">{avg_risk_prob:.1f}%</div>
                    <div class="risk-summary-label">평균 위험 확률</div>
                    <div class="risk-summary-sub">{len(risk_results)}개 자산 분석</div>
                </div>
                <div class="risk-card">
                    <h4>위험 수준 분포</h4>
                    <div class="risk-level-item">
                        <span class="risk-dot" style="background: #ef4444;"></span>
                        <span class="risk-label">CRITICAL (≥70%)</span>
                        <span class="risk-count">{risk_critical_count}</span>
                    </div>
                    <div class="risk-level-item">
                        <span class="risk-dot" style="background: #f97316;"></span>
                        <span class="risk-label">HIGH (50-70%)</span>
                        <span class="risk-count">{risk_high_count}</span>
                    </div>
                    <div class="risk-level-item">
                        <span class="risk-dot" style="background: #eab308;"></span>
                        <span class="risk-label">MEDIUM (30-50%)</span>
                        <span class="risk-count">{risk_medium_count}</span>
                    </div>
                    <div class="risk-level-item">
                        <span class="risk-dot" style="background: #22c55e;"></span>
                        <span class="risk-label">LOW (&lt;30%)</span>
                        <span class="risk-count">{risk_low_count}</span>
                    </div>
                </div>
                <div class="risk-card">
                    <h4>고위험 자산 Top 5</h4>
                    {"".join([f'<div class="risk-asset-item"><span class="risk-ticker">{r["ticker"]}</span><span class="risk-prob" style="color: {("#ef4444" if r["risk_prob"]*100 >= 70 else "#f97316" if r["risk_prob"]*100 >= 50 else "#eab308")};">{r["risk_prob"]*100:.1f}%</span></div>' for r in sorted(risk_results, key=lambda x: x.get("risk_prob", 0), reverse=True)[:5]]) if risk_results else '<p style="color: #9ca3af;">데이터 없음</p>'}
                </div>
            </div>
            
            <!-- 전체 자산별 위험확률 테이블 (NEW) -->
            <div class="risk-table-section">
                <h4>📊 전체 자산별 위험 확률</h4>
                <div class="risk-table-container">
                    <table class="risk-table">
                        <thead>
                            <tr>
                                <th>자산</th>
                                <th>위험확률</th>
                                <th>위험수준</th>
                                <th>모델</th>
                                <th>시각화</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'''<tr style="background: {'rgba(239,68,68,0.1)' if r.get('risk_prob',0)*100 >= 70 else 'rgba(249,115,22,0.1)' if r.get('risk_prob',0)*100 >= 50 else 'rgba(234,179,8,0.05)' if r.get('risk_prob',0)*100 >= 30 else 'transparent'};">
                                <td><strong>{r.get('ticker','')}</strong></td>
                                <td style="color: {'#ef4444' if r.get('risk_prob',0)*100 >= 70 else '#f97316' if r.get('risk_prob',0)*100 >= 50 else '#eab308' if r.get('risk_prob',0)*100 >= 30 else '#22c55e'}; font-weight: 600;">{r.get('risk_prob',0)*100:.1f}%</td>
                                <td><span class="level-badge" style="background: {'#ef4444' if r.get('risk_level')=='CRITICAL' else '#f97316' if r.get('risk_level')=='HIGH' else '#eab308' if r.get('risk_level')=='MEDIUM' else '#22c55e'};">{r.get('risk_level','N/A')}</span></td>
                                <td style="font-size: 0.75rem; color: #9ca3af;">{r.get('model_type','').replace('logistic_','').replace('_pooled','(P)')}</td>
                                <td><div class="risk-bar-container"><div class="risk-bar" style="width: {min(r.get('risk_prob',0)*100, 100):.0f}%; background: {'#ef4444' if r.get('risk_prob',0)*100 >= 70 else '#f97316' if r.get('risk_prob',0)*100 >= 50 else '#eab308' if r.get('risk_prob',0)*100 >= 30 else '#22c55e'};"></div></div></td>
                            </tr>''' for r in sorted(risk_results, key=lambda x: x.get('risk_prob', 0), reverse=True)])}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="risk-note">
                <p>📊 <strong>개별 자산 위험:</strong> 각 자산의 ML 위험 확률은 향후 10 거래일 내 5% 이상 하락(Max Drawdown) 가능성을 예측합니다.</p>
                <p>📈 <strong>전체 위험도:</strong> Critical Path 전체 위험도는 유동성, 신용, 변동성, 암호화폐 등 6개 경로의 가중평균으로 산출됩니다. 경로별 위험 점수에 시장 국면(레짐)별 가중치를 적용하여 0-100% 스케일로 표시합니다.</p>
                <p>이동평균(추세), 거래량(확신), 변동성(불확실성), VIX-실현변동성 괴리 등 지표를 종합 분석합니다.</p>
                <p style="color: #60a5fa; margin-top: 8px;">💡 <strong>모델 유형:</strong> equity/bond/commodity/crypto = 자산클래스별 모델, (P) = Pooled 모델</p>
            </div>
            
            <!-- 투자 성과 지표 (Risk-Adjusted Return Metrics) -->
            {_generate_risk_metrics_section(risk_metrics) if risk_metrics else ''}
        </div>''' if risk_enabled and risk_results else ''}
        
        <!-- AI 분석 요약 섹션 -->
        {f'''
        <!-- LLM 기반 AI 분석 요약 (Claude) -->
        {llm_summary_html}
        ''' if llm_summary_html else f'''
        <!-- 기존 AI 요약 (LLM 요약 없을 때 표시) -->
        <div class="summary-section">
            <h3>🤖 AI 시장 분석</h3>
            <div class="summary-content">
                {summary_html}
            </div>
        </div>
        '''}
        
        <!-- 신호 테이블 -->
        <div class="table-section">
            <h3>🎯 실시간 신호 목록</h3>
            <table>
                <thead>
                    <tr>
                        <th>자산명</th>
                        <th>지표</th>
                        <th>레벨</th>
                        <th>🤖 ML Risk</th>
                        <th>Z-Score</th>
                        <th>설명</th>
                        <th>💡 대응 가이드</th>
                        <th>시간</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Signal News 섹션 -->
        {signal_news_html if signal_news_html else ''}
        
        <!-- 해석 섹션 -->
        {f'''<div class="interpretation-section">
            <h3>🔍 상세 해석</h3>
            {interpretation_html}
        </div>''' if interpretation_html else ''}
        
        <!-- 뉴스 섹션 -->
        {f'''<div class="news-section">
            <h3>📰 관련 뉴스</h3>
            {news_html}
        </div>''' if news_html else ''}
    </div>
    
    <script>
        // Chart.js 로드 확인 함수
        function checkChartAvailability() {{
            if (typeof window.Chart === 'undefined' || window.chartJsLoadFailed) {{
                return false;
            }}
            return true;
        }}
        
        // Fallback 테이블 렌더링 함수
        function renderFallbackTable(canvasId, data, chartType) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            const container = canvas.parentElement;
            const tableId = canvasId + '-fallback';
            
            // 기존 fallback 테이블이 있으면 제거
            const existingTable = document.getElementById(tableId);
            if (existingTable) {{
                existingTable.remove();
            }}
            
            // 테이블 생성
            const table = document.createElement('table');
            table.id = tableId;
            table.style.width = '100%';
            table.style.borderCollapse = 'collapse';
            table.style.marginTop = '20px';
            table.style.color = '#e4e4e7';
            table.style.fontSize = '0.9rem';
            
            const thead = document.createElement('thead');
            const tbody = document.createElement('tbody');
            
            if (chartType === 'doughnut') {{
                // 도넛 차트용 테이블
                const headerRow = document.createElement('tr');
                headerRow.innerHTML = '<th style="padding: 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.2);">레벨</th><th style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.2);">개수</th><th style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.2);">비율</th>';
                thead.appendChild(headerRow);
                
                const total = data.values.reduce((a, b) => a + b, 0);
                data.labels.forEach((label, idx) => {{
                    const row = document.createElement('tr');
                    const value = data.values[idx];
                    const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                    const color = data.colors[idx] || '#9ca3af';
                    
                    row.innerHTML = `
                        <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span style="display: inline-block; width: 12px; height: 12px; background: ${{color}}; border-radius: 2px; margin-right: 8px;"></span>
                            ${{label}}
                        </td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.1);">${{value}}</td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.1);">${{percentage}}%</td>
                    `;
                    tbody.appendChild(row);
                }});
            }} else if (chartType === 'bar') {{
                // 바 차트용 테이블
                const headerRow = document.createElement('tr');
                headerRow.innerHTML = '<th style="padding: 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.2);">지표</th><th style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.2);">신호 수</th>';
                thead.appendChild(headerRow);
                
                const sortedData = data.labels.map((label, idx) => ({{
                    label: label,
                    value: data.values[idx]
                }})).sort((a, b) => b.value - a.value);
                
                sortedData.forEach(item => {{
                    const row = document.createElement('tr');
                    const barWidth = Math.max((item.value / Math.max(...data.values)) * 100, 5);
                    row.innerHTML = `
                        <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);">${{item.label}}</td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                                <div style="flex: 1; max-width: 200px; height: 20px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                                    <div style="width: ${{barWidth}}%; height: 100%; background: #60a5fa; transition: width 0.3s;"></div>
                                </div>
                                <span style="min-width: 30px; text-align: right;">${{item.value}}</span>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(row);
                }});
            }}
            
            table.appendChild(thead);
            table.appendChild(tbody);
            container.appendChild(table);
            
            // 캔버스 숨기기
            canvas.style.display = 'none';
        }}
        
        // 에러 핸들링 함수
        function handleChartError(canvasId, error, chartType, data) {{
            console.error(`차트 생성 실패 (${{canvasId}}):`, error);
            
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            const container = canvas.parentElement;
            
            // 에러 메시지 표시
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = 'padding: 20px; text-align: center; color: #fca5a5; background: rgba(239, 68, 68, 0.1); border-radius: 8px; margin-top: 10px;';
            errorDiv.innerHTML = `
                <p style="margin: 0 0 8px 0; font-weight: 600;">⚠️ 차트 로드 실패</p>
                <p style="margin: 0; font-size: 0.85rem; color: #9ca3af;">데이터를 테이블로 표시합니다.</p>
            `;
            container.appendChild(errorDiv);
            
            // Fallback 테이블 렌더링
            if (data) {{
                renderFallbackTable(canvasId, data, chartType);
            }}
        }}
        
        // Signal News 확장/축소 함수
        function toggleNews(idx) {{
            const content = document.getElementById('news-content-' + idx);
            const toggle = event.currentTarget.querySelector('.signal-news-toggle');
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                toggle.textContent = '▲';
            }} else {{
                content.style.display = 'none';
                toggle.textContent = '▼';
            }}
        }}
        
        // 신호 분포 도넛 차트
        document.addEventListener('DOMContentLoaded', function() {{
            // Chart.js 로드 확인
            if (!checkChartAvailability()) {{
                console.warn('Chart.js를 사용할 수 없습니다. Fallback 모드로 전환합니다.');
                
                // 신호 분포 차트 Fallback
                const signalChartData = {{
                    labels: ['Critical', 'Alert', 'Warning'],
                    values: [{critical_count}, {alert_count}, {warning_count}],
                    colors: ['#ef4444', '#f97316', '#eab308']
                }};
                renderFallbackTable('signalChart', signalChartData, 'doughnut');
                
                // 지표별 분포 차트 Fallback
                const indicatorCounts = {{}};
                const signals = {json.dumps([s.get('indicator', '') for s in signals], ensure_ascii=False)};
                signals.forEach(ind => {{
                    indicatorCounts[ind] = (indicatorCounts[ind] || 0) + 1;
                }});
                
                const indicatorChartData = {{
                    labels: Object.keys(indicatorCounts),
                    values: Object.values(indicatorCounts)
                }};
                renderFallbackTable('indicatorChart', indicatorChartData, 'bar');
                
                return;
            }}
            
            // 신호 분포 도넛 차트 생성
            try {{
                const signalCtx = document.getElementById('signalChart');
                if (!signalCtx) {{
                    console.warn('signalChart 캔버스를 찾을 수 없습니다.');
                    return;
                }}
                
                const signalChart = new Chart(signalCtx.getContext('2d'), {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Critical', 'Alert', 'Warning'],
                        datasets: [{{
                            data: [{critical_count}, {alert_count}, {warning_count}],
                            backgroundColor: ['#ef4444', '#f97316', '#eab308'],
                            borderColor: ['#dc2626', '#ea580c', '#ca8a04'],
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ color: '#e4e4e7' }}
                            }}
                        }}
                    }}
                }});
            }} catch (error) {{
                const signalChartData = {{
                    labels: ['Critical', 'Alert', 'Warning'],
                    values: [{critical_count}, {alert_count}, {warning_count}],
                    colors: ['#ef4444', '#f97316', '#eab308']
                }};
                handleChartError('signalChart', error, 'doughnut', signalChartData);
            }}
            
            // 지표별 분포 바 차트 생성
            try {{
                const indicatorCounts = {{}};
                const signals = {json.dumps([s.get('indicator', '') for s in signals], ensure_ascii=False)};
                signals.forEach(ind => {{
                    indicatorCounts[ind] = (indicatorCounts[ind] || 0) + 1;
                }});
                
                const indicatorCtx = document.getElementById('indicatorChart');
                if (!indicatorCtx) {{
                    console.warn('indicatorChart 캔버스를 찾을 수 없습니다.');
                    return;
                }}
                
                const indicatorChart = new Chart(indicatorCtx.getContext('2d'), {{
                    type: 'bar',
                    data: {{
                        labels: Object.keys(indicatorCounts),
                        datasets: [{{
                            label: '신호 수',
                            data: Object.values(indicatorCounts),
                            backgroundColor: '#60a5fa',
                            borderColor: '#3b82f6',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                ticks: {{ color: '#9ca3af' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            x: {{
                                ticks: {{ color: '#9ca3af' }},
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }} catch (error) {{
                const indicatorCounts = {{}};
                const signals = {json.dumps([s.get('indicator', '') for s in signals], ensure_ascii=False)};
                signals.forEach(ind => {{
                    indicatorCounts[ind] = (indicatorCounts[ind] || 0) + 1;
                }});
                
                const indicatorChartData = {{
                    labels: Object.keys(indicatorCounts),
                    values: Object.values(indicatorCounts)
                }};
                handleChartError('indicatorChart', error, 'bar', indicatorChartData);
            }}
            
            // Markov Regime 확률 시계열 차트 생성
            {_generate_markov_charts_js(regime_data.get('markov_analysis', {})) if regime_data.get('markov_analysis') else ''}
        }});
    </script>
</body>
</html>"""
    
    return html


if __name__ == "__main__":
    # 테스트
    test_signals = [
        {"ticker": "GC=F", "name": "Gold Futures", "indicator": "volume", "level": "CRITICAL", "z_score": 4.24, "description": "거래량 급증", "action_guide": "방향성 확인 후 대응", "timestamp": "2025-12-04T18:42:08"},
        {"ticker": "HG=F", "name": "Copper", "indicator": "return_z", "level": "CRITICAL", "z_score": 3.39, "description": "수익률 이상", "action_guide": "추격 매수 자제", "timestamp": "2025-12-04T18:42:08"},
    ]
    
    html = generate_dashboard(test_signals, "테스트 요약", [], [])
    
    with open("test_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("Dashboard generated: test_dashboard.html")