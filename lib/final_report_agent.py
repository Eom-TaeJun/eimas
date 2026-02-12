#!/usr/bin/env python3
"""
Final Report Agent v2.0
========================
EIMAS 분석 결과를 종합하여 최종 HTML 리포트를 생성하는 에이전트.

outputs/의 최신 JSON/MD 파일을 읽어 포괄적인 투자 보고서를 생성합니다.
경제/금융 도메인 지식 기반으로 시각화와 분석을 제공합니다.

v2.0 업데이트:
- JSON 데이터 전체 반영 (HFT, GARCH, PoI, Reasoning Chain, 등)
- MD 데이터 전체 반영 (기술적 지표, 국제 시장, 섹터 분석 등)
- 새로운 금융 에이전트 결과 포함

사용법:
    python -m lib.final_report_agent --user "엄태준"
    python -m lib.final_report_agent --output ./custom_reports/
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import modularized components
from lib.reports.themes import CSS_LIGHT_THEME as CSS_THEME
from lib.reports.charts import generate_svg_pie_chart


class FinalReportAgent:
    """
    경제/금융 도메인 최종 리포트 생성 에이전트 v2.0

    outputs/ 디렉토리의 최신 분석 결과를 읽어
    포괄적인 HTML 리포트를 생성합니다.

    v2.0: 모든 JSON/MD 데이터 반영
    """

    def __init__(self, output_dir: str = "outputs", user_name: str = "EIMAS"):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.user_name = user_name

        # Data holders
        self.integrated_data: Dict = {}
        self.ai_report_content: str = ""
        self.ai_report_sections: Dict = {}
        self.ai_report_raw: Dict = {}
        self.ib_memo_content: str = ""
        self.timestamp = datetime.now()

    def _get_latest_file(self, pattern: str) -> Optional[Path]:
        """지정된 패턴의 최신 파일 반환"""
        files = list(self.output_dir.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)

    def _resolve_output_path(self, path_str: str) -> Optional[Path]:
        """상대/절대 경로를 output_dir 기준으로 안전하게 해석"""
        if not path_str:
            return None

        candidate = Path(path_str)
        if candidate.exists():
            return candidate

        # report_path가 outputs/xxx 형태일 수 있어 파일명 기준으로 재시도
        candidate_by_name = self.output_dir / candidate.name
        if candidate_by_name.exists():
            return candidate_by_name

        return None

    def load_latest_data(self) -> Dict:
        """outputs/에서 최신 JSON/MD 파일 로드"""
        # 1. Load unified EIMAS JSON (NEW: eimas_*.json)
        json_file = self._get_latest_file("eimas_*.json")
        if not json_file:
            # Fallback to old format
            json_file = self._get_latest_file("integrated_*.json")
        
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.integrated_data = json.load(f)
            print(f"  [OK] Loaded: {json_file.name}")
        else:
            print("  [WARN] No eimas_*.json or integrated_*.json found")

        # 2. AI report 섹션/원본 로드 (JSON + MD 보강)
        ai_report = self.integrated_data.get('ai_report') or {}
        self.ai_report_sections = {}
        self.ai_report_raw = {}

        if isinstance(ai_report, dict):
            unified_sections = ai_report.get('sections') or {}
            if isinstance(unified_sections, dict):
                self.ai_report_sections = dict(unified_sections)
            raw_from_unified = ai_report.get('report_data') or {}
            if isinstance(raw_from_unified, dict):
                self.ai_report_raw = dict(raw_from_unified)

        ai_md_file: Optional[Path] = None
        ai_json_file: Optional[Path] = None

        # report_path가 있으면 동일 타임스탬프의 md/json을 우선 사용
        if isinstance(ai_report, dict):
            report_path = ai_report.get('report_path', '')
            resolved_md = self._resolve_output_path(report_path)
            if resolved_md and resolved_md.suffix.lower() == '.md':
                ai_md_file = resolved_md
                resolved_json = resolved_md.with_suffix('.json')
                if resolved_json.exists():
                    ai_json_file = resolved_json

        if ai_md_file is None:
            ai_md_file = self._get_latest_file("ai_report_*.md")
        if ai_json_file is None:
            ai_json_file = self._get_latest_file("ai_report_*.json")

        parsed_sections: Dict[str, Dict] = {}
        if ai_md_file:
            with open(ai_md_file, 'r', encoding='utf-8') as f:
                self.ai_report_content = f.read()
            parsed_sections = self._parse_md_sections(self.ai_report_content)
            print(f"  [OK] Loaded: {ai_md_file.name} ({len(parsed_sections)} parsed sections)")

        # 통합 JSON 섹션 + MD 파싱 섹션 병합
        if parsed_sections:
            for key, value in parsed_sections.items():
                existing = self.ai_report_sections.get(key)
                existing_content = existing.get('content', '') if isinstance(existing, dict) else ''
                new_content = value.get('content', '') if isinstance(value, dict) else ''
                if (not existing) or (len(new_content) > len(existing_content)):
                    self.ai_report_sections[key] = value

        if self.ai_report_sections:
            print(f"  [OK] AI Report sections ready ({len(self.ai_report_sections)} sections)")
        else:
            print("  [WARN] No AI Report sections found")

        # AI 리포트 원본(JSON) 로드
        if ai_json_file:
            try:
                with open(ai_json_file, 'r', encoding='utf-8') as f:
                    loaded_raw = json.load(f)
                if isinstance(loaded_raw, dict):
                    # 통합 report_data가 있을 때는 raw에서 빈 필드만 보강
                    for key, value in loaded_raw.items():
                        if key not in self.ai_report_raw or not self.ai_report_raw.get(key):
                            self.ai_report_raw[key] = value
                    print(f"  [OK] Loaded: {ai_json_file.name} (raw ai report)")
            except Exception as e:
                print(f"  [WARN] Failed to load ai_report json: {e}")

        # 3. Load IB memo MD (legacy)
        ib_file = self._get_latest_file("ib_memorandum_*.md")
        if ib_file:
            with open(ib_file, 'r', encoding='utf-8') as f:
                self.ib_memo_content = f.read()
            print(f"  [OK] Loaded: {ib_file.name}")

        return {
            "integrated": self.integrated_data,
            "ai_sections": self.ai_report_sections,
            "ai_raw": self.ai_report_raw,
            "ib_memo": self.ib_memo_content
        }

    def _parse_md_sections(self, content: str) -> Dict[str, Dict]:
        """## N. Section Title 패턴으로 섹션 추출"""
        sections = {}
        # Match ## followed by number and title
        pattern = r'## (\d+)\. (.+?)\n(.*?)(?=\n## \d+\.|$)'
        for match in re.finditer(pattern, content, re.DOTALL):
            num, title, body = match.groups()
            sections[f"section_{num}"] = {
                "title": title.strip(),
                "content": body.strip()
            }
        return sections

    def _safe_get(self, data: Dict, *keys, default=None):
        """안전하게 중첩 딕셔너리에서 값 추출"""
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key, default)
            else:
                return default
        return result if result is not None else default

    def generate_report(self) -> str:
        """전체 HTML 리포트 생성"""
        html_parts = [
            self._generate_head(),
            '<body>',
            '<div class="container">',
            self._generate_header(),
            self._generate_change_comparison(),       # NEW: 이전 대비 변화
            self._generate_executive_summary(),
            self._generate_extended_metrics(),        # NEW: 확장 지표
            self._generate_institutional_frameworks(), # NEW: JP Morgan/Goldman Sachs 프레임워크
            self._generate_valuation_section(),
            self._generate_technical_indicators(),    # NEW: 기술적 지표 (RSI, MACD 등)
            self._generate_global_markets(),          # NEW: 국제 시장
            self._generate_ark_invest_section(),
            self._generate_market_structure_section(),
            self._generate_volume_shock_section(),    # NEW: 거래량 이상징후 & 충격 전파
            self._generate_hft_microstructure(),      # NEW: HFT 상세
            self._generate_garch_volatility(),        # NEW: GARCH 상세
            self._generate_information_flow(),        # NEW: CAPM, 이상거래
            self._generate_proof_of_index(),          # NEW: PoI 상세
            self._generate_debate_section(),
            self._generate_institutional_narrative(), # NEW: 기관 투자자 분석 내러티브
            self._generate_ai_institutional_interpretation(), # NEW: AI 기관 분석 해석
            self._generate_school_interpretations(),  # NEW: 학파별 해석
            self._generate_reasoning_chain(),         # NEW: 추론 과정
            self._generate_portfolio_section(),
            self._generate_adaptive_portfolios(),     # NEW: 적응형 포트폴리오
            self._generate_sector_analysis(),         # NEW: 섹터 분석
            self._generate_entry_exit_section(),
            self._generate_watchlist_section(),       # NEW: 주목할 종목
            self._generate_news_section(),
            self._generate_scenario_section(),
            self._generate_final_proposal(),          # NEW: 최종 제안
            self._generate_operational_decision(),    # NEW: 운용 의사결정 시스템
            self._generate_ai_analysis_section(),
            self._generate_footer(),
            '</div>',
            '</body>',
            '</html>'
        ]
        return '\n'.join(html_parts)

    def save_report(self) -> Path:
        """outputs/reports/에 저장"""
        html = self.generate_report()
        date_str = self.timestamp.strftime("%Y%m%d")
        filename = f"{self.user_name}_report_summary_{date_str}.html"
        output_path = self.reports_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n[SUCCESS] Report saved: {output_path}")
        return output_path

    def save_pdf(self, html_path: Path = None) -> Path:
        """HTML을 PDF로 변환 (wkhtmltopdf 필요)"""
        import subprocess
        
        if html_path is None:
            html_path = self.save_report()
        
        pdf_path = html_path.with_suffix('.pdf')
        
        try:
            result = subprocess.run([
                'wkhtmltopdf',
                '--enable-local-file-access',
                '--encoding', 'utf-8',
                '--page-size', 'A4',
                '--margin-top', '10mm',
                '--margin-bottom', '10mm',
                '--margin-left', '10mm',
                '--margin-right', '10mm',
                str(html_path),
                str(pdf_path)
            ], capture_output=True, text=True, timeout=60)
            
            if pdf_path.exists():
                print(f"[SUCCESS] PDF saved: {pdf_path}")
                return pdf_path
            else:
                print(f"[WARN] PDF conversion failed: {result.stderr}")
                return None
        except FileNotFoundError:
            print("[WARN] wkhtmltopdf not installed. Install with: sudo apt install wkhtmltopdf")
            return None
        except subprocess.TimeoutExpired:
            print("[WARN] PDF conversion timeout")
            return None

    # ========================================================================
    # Section Generators
    # ========================================================================

    def _generate_head(self) -> str:
        """HTML head + CSS"""
        return f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIMAS 통합 전략 보고서 - {self.user_name}</title>
    <style>
{CSS_THEME}
    </style>
</head>'''

    def _generate_header(self) -> str:
        """헤더 섹션"""
        data = self.integrated_data
        timestamp = data.get('timestamp', self.timestamp.isoformat())
        recommendation = data.get('final_recommendation', 'NEUTRAL')

        rec_lower = recommendation.lower()
        if 'bull' in rec_lower or 'buy' in rec_lower:
            badge_class = 'bullish'
            badge_text = 'BULLISH (매수 권장)'
        elif 'bear' in rec_lower or 'sell' in rec_lower:
            badge_class = 'bearish'
            badge_text = 'BEARISH (매도 권장)'
        else:
            badge_class = 'neutral'
            badge_text = 'NEUTRAL (중립)'

        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y년 %m월 %d일 %H:%M")
        except:
            formatted_time = timestamp[:16] if len(timestamp) > 16 else timestamp

        return f'''
<div class="header">
    <div>
        <h1>EIMAS 통합 전략 보고서</h1>
        <p class="meta">생성일시: {formatted_time} | 버전: v2.2.0</p>
    </div>
    <div class="status-badge {badge_class}">{badge_text}</div>
</div>'''

    def _generate_change_comparison(self) -> str:
        """이전 리포트 대비 변화 - 현재 분석 요약으로 대체"""
        data = self.integrated_data

        # 현재 데이터 추출
        regime = data.get('regime', {})
        regime_type = regime.get('regime', 'Unknown') if isinstance(regime, dict) else str(regime)
        trend = regime.get('trend', 'N/A') if isinstance(regime, dict) else 'N/A'
        volatility = regime.get('volatility', 'N/A') if isinstance(regime, dict) else 'N/A'

        confidence = data.get('confidence', 0)
        if confidence <= 1:
            confidence *= 100

        risk_score = data.get('risk_score', 0)
        recommendation = data.get('final_recommendation', 'N/A')

        # 리스크 레벨 판단
        if risk_score < 30:
            risk_level, risk_class = '낮음', 'text-green'
        elif risk_score < 50:
            risk_level, risk_class = '중간', 'text-yellow'
        else:
            risk_level, risk_class = '높음', 'text-red'

        # 권고 색상
        if 'BULL' in recommendation.upper():
            rec_class = 'text-green'
        elif 'BEAR' in recommendation.upper():
            rec_class = 'text-red'
        else:
            rec_class = 'text-yellow'

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-purple);">
    <div class="card-header">
        <span class="card-title">📊 현재 분석 요약</span>
        <span class="metric-badge bg-blue">실시간</span>
    </div>
    <table>
        <tr>
            <th>항목</th>
            <th>현재 값</th>
            <th>상태</th>
        </tr>
        <tr>
            <td>시장 레짐</td>
            <td><strong>{regime_type}</strong></td>
            <td>추세: {trend}, 변동성: {volatility}</td>
        </tr>
        <tr>
            <td>AI 신뢰도</td>
            <td><strong>{confidence:.0f}%</strong></td>
            <td>{'높음' if confidence >= 70 else '중간' if confidence >= 50 else '낮음'}</td>
        </tr>
        <tr>
            <td>리스크 점수</td>
            <td><strong class="{risk_class}">{risk_score:.1f}</strong></td>
            <td>{risk_level}</td>
        </tr>
        <tr>
            <td>투자 권고</td>
            <td><strong class="{rec_class}">{recommendation}</strong></td>
            <td>-</td>
        </tr>
    </table>
</div>'''

    def _generate_executive_summary(self) -> str:
        """핵심 지표 요약 (5개 카드)"""
        data = self.integrated_data
        risk_score = data.get('risk_score', 0)
        confidence = data.get('confidence', 0) * 100 if data.get('confidence', 0) <= 1 else data.get('confidence', 0)
        regime = data.get('regime', {})

        regime_type = regime.get('regime', 'Unknown') if isinstance(regime, dict) else str(regime)
        regime_conf = regime.get('confidence', 0.75) if isinstance(regime, dict) else 0.75
        if regime_conf <= 1:
            regime_conf *= 100

        fred = data.get('fred_summary', {})
        net_liq = fred.get('net_liquidity', 0)
        net_liq_display = f"${net_liq/1e3:.1f}T" if net_liq > 1000 else f"${net_liq:.0f}B"
        liq_regime = fred.get('liquidity_regime', 'Abundant')

        # Risk level
        if risk_score < 30:
            risk_color, risk_level = 'text-green', '매우 낮음'
        elif risk_score < 50:
            risk_color, risk_level = 'text-blue', '낮음'
        elif risk_score < 70:
            risk_color, risk_level = 'text-yellow', '중간'
        else:
            risk_color, risk_level = 'text-red', '높음'

        return f'''
<div class="grid grid-5">
    <div class="card">
        <p class="metric-label">리스크 점수</p>
        <p class="metric-value-large {risk_color}">{risk_score:.1f}</p>
        <span class="metric-badge bg-{'green' if risk_score < 30 else 'yellow' if risk_score < 70 else 'red'}">{risk_level}</span>
    </div>
    <div class="card">
        <p class="metric-label">시장 레짐</p>
        <p class="metric-value-medium text-blue">{regime_type}</p>
        <p class="text-muted" style="font-size: 0.85rem;">신뢰도 {regime_conf:.0f}%</p>
    </div>
    <div class="card">
        <p class="metric-label">AI 신뢰도</p>
        <p class="metric-value-large text-purple">{confidence:.0f}%</p>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {confidence}%; background: var(--accent-purple);"></div>
        </div>
    </div>
    <div class="card">
        <p class="metric-label">순유동성</p>
        <p class="metric-value-large text-cyan">{net_liq_display}</p>
        <span class="metric-badge bg-cyan">{liq_regime}</span>
    </div>
    <div class="card">
        <p class="metric-label">Fed Funds Rate</p>
        <p class="metric-value-large">{fred.get('fed_funds', 0):.2f}%</p>
        <p class="text-muted" style="font-size: 0.85rem;">10Y: {fred.get('treasury_10y', 0):.2f}%</p>
    </div>
</div>'''

    def _generate_extended_metrics(self) -> str:
        """확장 시장 지표 (NEW)"""
        data = self.integrated_data
        ext = data.get('extended_data', {})

        pcr = self._safe_get(ext, 'put_call_ratio', 'ratio', default=1.0)
        pcr_sentiment = self._safe_get(ext, 'put_call_ratio', 'sentiment', default='NEUTRAL')

        fundamentals = ext.get('fundamentals', {})
        pe_ratio = fundamentals.get('pe_ratio', 28)
        earnings_yield = fundamentals.get('earnings_yield', 3.5)

        digital_liq = ext.get('digital_liquidity', {})
        stable_mcap = digital_liq.get('total_mcap', 0)
        stable_mcap_display = f"${stable_mcap/1e9:.1f}B" if stable_mcap > 0 else "N/A"

        credit = ext.get('credit_spreads', {})
        risk_ratio = credit.get('risk_ratio_hyg_ief', 0.85)
        credit_interp = credit.get('interpretation', 'Risk OFF')

        # PCR color
        if pcr > 1.2:
            pcr_color, pcr_badge = 'text-red', 'bg-red'
        elif pcr < 0.8:
            pcr_color, pcr_badge = 'text-green', 'bg-green'
        else:
            pcr_color, pcr_badge = 'text-yellow', 'bg-yellow'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📈 확장 시장 지표</span>
    </div>
    <div class="grid grid-4" style="margin-bottom: 0;">
        <div class="tech-item">
            <p class="tech-label">Put/Call Ratio</p>
            <p class="tech-value {pcr_color}">{pcr:.2f}</p>
            <span class="tech-badge {pcr_badge}">{pcr_sentiment}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">S&P 500 P/E</p>
            <p class="tech-value">{pe_ratio:.1f}x</p>
            <span class="tech-badge bg-blue">Earnings Yield {earnings_yield:.2f}%</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">Stablecoin MCap</p>
            <p class="tech-value text-purple">{stable_mcap_display}</p>
            <span class="tech-badge bg-purple">Digital Liquidity</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">Credit Spreads</p>
            <p class="tech-value">{risk_ratio:.3f}</p>
            <span class="tech-badge {'bg-red' if 'OFF' in credit_interp else 'bg-green'}">{credit_interp}</span>
        </div>
    </div>
</div>'''

    def _generate_institutional_frameworks(self) -> str:
        """기관급 분석 프레임워크 (JP Morgan, Goldman Sachs) - NEW 2026-01-31"""
        data = self.integrated_data

        bubble_fw = data.get('bubble_framework', {})
        gap_analysis = data.get('gap_analysis', {})
        fomc = data.get('fomc_analysis', {})

        # 데이터가 없으면 빈 문자열 반환
        if not bubble_fw and not gap_analysis and not fomc:
            return ''

        # 5-Stage Bubble Framework
        bubble_html = ''
        if bubble_fw:
            stage = bubble_fw.get('stage', 'N/A')
            score = bubble_fw.get('total_score', 0)
            stage_results = bubble_fw.get('stage_results', [])

            # 단계별 색상
            stage_colors = {
                'NO_BUBBLE': ('text-green', 'bg-green'),
                'EARLY_FORMATION': ('text-yellow', 'bg-yellow'),
                'BUBBLE_BUILDING': ('text-orange', 'bg-orange'),
                'LATE_STAGE': ('text-red', 'bg-red'),
                'IMMINENT_POP': ('text-red', 'bg-red')
            }
            color, badge_color = stage_colors.get(stage, ('text-yellow', 'bg-yellow'))

            # 단계별 상세
            stages_html = ''
            for sr in stage_results:
                s_name = sr.get('stage', '').replace('_', ' ').title()
                s_passed = sr.get('passed', True)
                s_score = sr.get('score', 0)
                s_evidence = sr.get('evidence', '')[:60]
                icon = '✅' if s_passed else '⚠️'
                stages_html += f'''
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span>{icon} {s_name}</span>
                    <span style="color: var(--text-secondary);">{s_score:.0f}/20</span>
                </div>'''

            bubble_html = f'''
            <div class="tech-item" style="flex: 1; min-width: 280px;">
                <p class="tech-label">5-Stage Bubble (JP Morgan WM)</p>
                <p class="tech-value {color}">{stage.replace('_', ' ')}</p>
                <span class="tech-badge {badge_color}">Score: {score:.0f}/100</span>
                <div style="margin-top: 12px; font-size: 0.85rem;">
                    {stages_html}
                </div>
            </div>'''

        # Gap Analysis
        gap_html = ''
        if gap_analysis:
            signal = gap_analysis.get('overall_signal', 'NEUTRAL')
            opportunity = gap_analysis.get('opportunity', '')[:80]
            pessimistic = gap_analysis.get('market_too_pessimistic', False)
            optimistic = gap_analysis.get('market_too_optimistic', False)
            confidence = gap_analysis.get('confidence', 0.5)

            if signal == 'BULLISH':
                color, badge_color = 'text-green', 'bg-green'
            elif signal == 'BEARISH':
                color, badge_color = 'text-red', 'bg-red'
            else:
                color, badge_color = 'text-yellow', 'bg-yellow'

            gaps = gap_analysis.get('gaps', [])
            gaps_html = ''
            for g in gaps:
                metric = g.get('metric', '').replace('_', ' ').title()
                g_signal = g.get('signal', 'NEUTRAL')
                implied = g.get('market_implied', 0)
                forecast = g.get('model_forecast', 0)
                g_icon = '📈' if g_signal == 'BULLISH' else ('📉' if g_signal == 'BEARISH' else '➖')
                gaps_html += f'''
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem;">
                    <span>{g_icon} {metric}</span>
                    <span>{implied:.1f} vs {forecast:.1f}</span>
                </div>'''

            gap_html = f'''
            <div class="tech-item" style="flex: 1; min-width: 280px;">
                <p class="tech-label">Market-Model Gap (Goldman Sachs)</p>
                <p class="tech-value {color}">{signal}</p>
                <span class="tech-badge {badge_color}">Confidence: {confidence:.0%}</span>
                <p style="margin-top: 8px; color: var(--text-secondary); font-size: 0.85rem;">{opportunity}</p>
                <div style="margin-top: 12px;">
                    {gaps_html}
                </div>
            </div>'''

        # FOMC Analysis
        fomc_html = ''
        if fomc:
            stance = fomc.get('stance', 'N/A')
            proj = fomc.get('2026_projections', {})
            median_rate = proj.get('median', 0)
            rate_range = proj.get('range', [0, 0])
            uncertainty = fomc.get('uncertainty', {})
            policy_unc = uncertainty.get('policy_uncertainty_index', 0)
            member_dist = fomc.get('member_distribution', {})

            if stance == 'HAWKISH':
                color, badge_color = 'text-red', 'bg-red'
            elif stance == 'DOVISH':
                color, badge_color = 'text-green', 'bg-green'
            else:
                color, badge_color = 'text-yellow', 'bg-yellow'

            # 시나리오 경로
            scenarios = fomc.get('scenarios', {})
            base_path = scenarios.get('base', [])
            hawkish_path = scenarios.get('hawkish', [])
            dovish_path = scenarios.get('dovish', [])

            path_html = ''
            if base_path:
                path_str = ' → '.join([f"{r:.2f}%" for r in base_path])
                path_html = f'<p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 8px;">Base: {path_str}</p>'

            fomc_html = f'''
            <div class="tech-item" style="flex: 1; min-width: 280px;">
                <p class="tech-label">FOMC Dot Plot (JP Morgan AM)</p>
                <p class="tech-value {color}">{stance}</p>
                <span class="tech-badge {badge_color}">2026 Median: {median_rate:.2f}%</span>
                <div style="margin-top: 12px; font-size: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span>Range</span>
                        <span>{rate_range[0]:.2f}% - {rate_range[1]:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span>Policy Uncertainty</span>
                        <span style="color: {'var(--accent-red)' if policy_unc > 50 else 'var(--accent-green)'};">{policy_unc:.0f}/100</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span>Hawkish / Neutral / Dovish</span>
                        <span>{member_dist.get('hawkish', 0)} / {member_dist.get('neutral', 0)} / {member_dist.get('dovish', 0)}</span>
                    </div>
                </div>
                {path_html}
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-purple);">
    <div class="card-header">
        <span class="card-title">🏛️ 기관급 분석 프레임워크 (JP Morgan / Goldman Sachs)</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 20px; padding: 16px;">
        {bubble_html}
        {gap_html}
        {fomc_html}
    </div>
</div>'''

    def _generate_valuation_section(self) -> str:
        """Fed Model 밸류에이션"""
        data = self.integrated_data
        fred = data.get('fred_summary', {})
        ext = data.get('extended_data', {})

        fundamentals = ext.get('fundamentals', {})
        earnings_yield = fundamentals.get('earnings_yield', 3.5)
        treasury_10y = fred.get('treasury_10y', 4.2)

        premium = earnings_yield - treasury_10y
        warning_html = f'''<div style="background: var(--accent-{'red' if premium < 0 else 'green'}-bg);
            color: var(--accent-{'red' if premium < 0 else 'green'}); padding: 10px; border-radius: 6px;
            margin-top: 12px; text-align: center;">
            <strong>{'경고: 주식이 채권보다 비쌈 (음의 프리미엄)' if premium < 0 else '주식이 상대적으로 매력적 (양의 프리미엄)'}</strong>
            <br>프리미엄: {premium:+.2f}%p
        </div>'''

        max_val = max(earnings_yield, treasury_10y, 5)
        eq_width = (earnings_yield / max_val) * 100
        bond_width = (treasury_10y / max_val) * 100

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">💰 밸류에이션 (Fed Model)</span>
    </div>
    <div class="valuation-row">
        <span class="valuation-label">주식 (SPX)</span>
        <div class="valuation-bar-container">
            <div class="valuation-bar" style="width: {eq_width}%; background: var(--accent-blue);"></div>
        </div>
        <span class="valuation-value text-blue">{earnings_yield:.2f}%</span>
    </div>
    <div class="valuation-row">
        <span class="valuation-label">채권 (10Y)</span>
        <div class="valuation-bar-container">
            <div class="valuation-bar" style="width: {bond_width}%; background: var(--accent-red);"></div>
        </div>
        <span class="valuation-value text-red">{treasury_10y:.2f}%</span>
    </div>
    {warning_html}
</div>'''

    def _generate_technical_indicators(self) -> str:
        """기술적 지표 (NEW) - RSI, MACD, 이동평균선 또는 대체 지표"""
        data = self.integrated_data

        # JSON에서 직접 추출 (fred_summary, extended_data)
        extended = data.get('extended_data', {})
        fred = data.get('fred_summary', {})
        sentiment = data.get('sentiment_analysis', {})
        vix_structure = sentiment.get('vix_structure', {})

        # MD 섹션 3에서 추출 (fallback)
        section = self.ai_report_sections.get('section_3', {})
        content = section.get('content', '')

        # 기본값 (parsing 실패 시)
        rsi = None
        macd = None
        macd_signal = None
        ma50 = None
        ma200 = None
        support = None
        resistance = None

        # RSI 추출
        rsi_match = re.search(r'RSI.*?(\d+\.?\d*)', content)
        if rsi_match:
            rsi = float(rsi_match.group(1))

        # MACD 추출
        macd_match = re.search(r'MACD.*?(\-?\d+\.?\d*)', content)
        if macd_match:
            macd = float(macd_match.group(1))

        macd_sig_match = re.search(r'MACD Signal.*?(\-?\d+\.?\d*)', content)
        if macd_sig_match:
            macd_signal = float(macd_sig_match.group(1))

        # MA 추출
        ma50_match = re.search(r'50일.*?(\d+\.?\d*)', content)
        if ma50_match:
            ma50 = float(ma50_match.group(1))

        ma200_match = re.search(r'200일.*?(\d+\.?\d*)', content)
        if ma200_match:
            ma200 = float(ma200_match.group(1))

        # Support/Resistance 추출
        supp_match = re.search(r'지지선.*?(\d+\,?\d*\.?\d*)', content)
        if supp_match:
            support = float(supp_match.group(1).replace(',', ''))

        res_match = re.search(r'저항선.*?(\d+\,?\d*\.?\d*)', content)
        if res_match:
            resistance = float(res_match.group(1).replace(',', ''))

        # MD에서 추출 실패 시 JSON 데이터로 대체 지표 표시
        all_none = all(v is None for v in [rsi, macd, ma50, ma200])
        if all_none:
            # VIX, Put/Call, Fear & Greed 등 대체 지표 사용
            vix_spot = vix_structure.get('vix_spot', 0)
            vix_signal = vix_structure.get('signal', 'N/A')
            vix_structure_type = vix_structure.get('structure', 'N/A')

            put_call = extended.get('put_call_ratio', {})
            pc_ratio = put_call.get('ratio', 0)
            pc_sentiment = put_call.get('sentiment', 'NEUTRAL')

            fear_greed = sentiment.get('fear_greed', {})
            fg_value = fear_greed.get('value', 50)
            fg_level = fear_greed.get('level', 'neutral')

            fundamentals = extended.get('fundamentals', {})
            pe_ratio = fundamentals.get('pe_ratio', 0)
            earnings_yield = fundamentals.get('earnings_yield', 0)

            # VIX 해석
            if vix_spot < 15:
                vix_class, vix_label = 'text-green', '낮음 (안정)'
            elif vix_spot < 25:
                vix_class, vix_label = 'text-blue', '보통'
            elif vix_spot < 35:
                vix_class, vix_label = 'text-yellow', '높음'
            else:
                vix_class, vix_label = 'text-red', '매우 높음 (공포)'

            # Put/Call 해석
            if pc_ratio < 0.7:
                pc_class = 'text-green'
            elif pc_ratio > 1.0:
                pc_class = 'text-red'
            else:
                pc_class = 'text-yellow'

            # Fear & Greed 해석
            if fg_value < 25:
                fg_class = 'text-red'
            elif fg_value > 75:
                fg_class = 'text-green'
            else:
                fg_class = 'text-yellow'

            return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📊 시장 센티먼트 지표</span>
        <span class="text-muted" style="font-size: 0.85rem;">기술적 지표 대체</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 16px;">
        <div class="tech-item">
            <p class="tech-label">VIX (변동성 지수)</p>
            <p class="tech-value {vix_class}">{vix_spot:.1f}</p>
            <span class="tech-badge bg-blue">{vix_label}</span>
            <p class="text-muted" style="font-size: 0.8rem; margin-top: 4px;">{vix_structure_type}</p>
        </div>
        <div class="tech-item">
            <p class="tech-label">Put/Call Ratio</p>
            <p class="tech-value {pc_class}">{pc_ratio:.2f}</p>
            <span class="tech-badge {'bg-red' if 'BEAR' in pc_sentiment else 'bg-green' if 'BULL' in pc_sentiment else 'bg-yellow'}">{pc_sentiment}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">Fear & Greed</p>
            <p class="tech-value {fg_class}">{fg_value}</p>
            <span class="tech-badge bg-blue">{fg_level.title()}</span>
        </div>
    </div>
    <div class="grid grid-2" style="margin-bottom: 0;">
        <div style="text-align: center; padding: 12px; background: var(--bg-tertiary); border-radius: 8px;">
            <p class="text-muted" style="font-size: 0.85rem;">P/E Ratio (S&P 500)</p>
            <p style="font-weight: 700; font-size: 1.2rem;">{pe_ratio:.1f}x</p>
        </div>
        <div style="text-align: center; padding: 12px; background: var(--bg-tertiary); border-radius: 8px;">
            <p class="text-muted" style="font-size: 0.85rem;">Earnings Yield</p>
            <p style="font-weight: 700; font-size: 1.2rem;">{earnings_yield:.2f}%</p>
        </div>
    </div>
</div>'''

        # RSI 해석
        if rsi is not None:
            if rsi > 70:
                rsi_interp, rsi_class = '과매수', 'text-red'
            elif rsi < 30:
                rsi_interp, rsi_class = '과매도', 'text-green'
            else:
                rsi_interp, rsi_class = '중립', 'text-blue'
            rsi_display = f"{rsi:.1f}"
        else:
            rsi_interp, rsi_class = 'N/A', 'text-muted'
            rsi_display = "N/A"

        # MACD 해석
        if macd is not None and macd_signal is not None:
            macd_badge = '매수 신호' if macd > macd_signal else '매도 신호'
            macd_bg = 'bg-green' if macd > macd_signal else 'bg-red'
            macd_display = f"{macd:.2f}"
        else:
            macd_badge = 'N/A'
            macd_bg = 'bg-gray'
            macd_display = f"{macd:.2f}" if macd is not None else "N/A"

        # 이동평균 상태
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:
                ma_status = '골든 크로스 (상승 추세)'
                ma_class = 'bg-green'
            else:
                ma_status = '데드 크로스 (하락 추세)'
                ma_class = 'bg-red'
        else:
            ma_status = 'N/A'
            ma_class = 'bg-gray'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📊 기술적 지표</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 16px;">
        <div class="tech-item">
            <p class="tech-label">RSI (14일)</p>
            <p class="tech-value {rsi_class}">{rsi_display}</p>
            <span class="tech-badge bg-blue">{rsi_interp}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">MACD</p>
            <p class="tech-value">{macd_display}</p>
            <span class="tech-badge {macd_bg}">{macd_badge}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">이동평균 상태</p>
            <p class="tech-value" style="font-size: 1rem;">{ "50MA > 200MA" if ma50 and ma200 and ma50 > ma200 else "N/A" }</p>
            <span class="tech-badge {ma_class}">{ma_status}</span>
        </div>
    </div>
    <div class="grid grid-4" style="margin-bottom: 0;">
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">50일 이동평균</p>
            <p style="font-weight: 700;">{f"${ma50:.2f}" if ma50 is not None else "N/A"}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">200일 이동평균</p>
            <p style="font-weight: 700;">{f"${ma200:.2f}" if ma200 is not None else "N/A"}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">지지선</p>
            <p style="font-weight: 700; color: var(--accent-green);">{f"${support:.2f}" if support is not None else "N/A"}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">저항선</p>
            <p style="font-weight: 700; color: var(--accent-red);">{f"${resistance:.2f}" if resistance is not None else "N/A"}</p>
        </div>
    </div>
</div>'''

    def _extract_market_data(self, content: str, key: str) -> tuple:
        """MD 콘텐츠에서 시장 데이터 추출 (가격, 변화율)"""
        # Pattern: - **Key**: Price (Change%)
        # Example: - **Gold**: $4,713.90 (-11.37%)
        # Example: - **DAX (독일)**: 24,538.81 (+0.94%)
        
        # Escape special chars in key if needed (e.g. ^VIX)
        escaped_key = re.escape(key)
        
        # Try finding line starting with - **Key
        pattern = fr'- \*\*{escaped_key}.*?\*\*:\s*([^\s]+)\s*\((.*?)\)'
        match = re.search(pattern, content)
        
        if match:
            price = match.group(1)
            change = match.group(2)
            
            # Determine color based on change
            if '-' in change:
                color = 'text-red'
            elif '+' in change:
                color = 'text-green'
            else:
                color = 'text-muted'
                
            return price, change, color
            
        return 'N/A', 'N/A', 'text-muted'

    def _generate_global_markets(self) -> str:
        """국제 시장 분석 (NEW)"""
        data = self.integrated_data

        # JSON에서 직접 추출 시도
        portfolio_weights = data.get('portfolio_weights', {})
        fred = data.get('fred_summary', {})

        # MD 섹션 4에서 추출 (fallback)
        section = self.ai_report_sections.get('section_4', {})
        content = section.get('content', '')

        # Global Indices
        dax_price, dax_chg, dax_col = self._extract_market_data(content, 'DAX')
        ftse_price, ftse_chg, ftse_col = self._extract_market_data(content, 'FTSE 100')
        nikkei_price, nikkei_chg, nikkei_col = self._extract_market_data(content, 'Nikkei 225')
        shanghai_price, shanghai_chg, shanghai_col = self._extract_market_data(content, 'Shanghai')
        kospi_price, kospi_chg, kospi_col = self._extract_market_data(content, 'KOSPI')

        # Commodities
        gold_price, gold_chg, gold_col = self._extract_market_data(content, 'Gold')
        wti_price, wti_chg, wti_col = self._extract_market_data(content, 'WTI 원유')
        copper_price, copper_chg, copper_col = self._extract_market_data(content, 'Copper')
        dxy_price, dxy_chg, dxy_col = self._extract_market_data(content, 'DXY')

        # 모든 데이터가 N/A인지 확인
        all_na = all(p == 'N/A' for p in [dax_price, ftse_price, nikkei_price, shanghai_price, kospi_price,
                                           gold_price, wti_price, copper_price, dxy_price])

        if all_na:
            # JSON에서 사용 가능한 데이터로 대체
            treasury_2y = fred.get('treasury_2y', 0)
            treasury_10y = fred.get('treasury_10y', 0)
            spread = fred.get('yield_spread_10y_2y', 0)
            fed_funds = fred.get('fed_funds', 0)

            return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🌍 글로벌 금리 및 유동성</span>
    </div>
    <div class="grid grid-2" style="margin-bottom: 0;">
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">미국 금리 구조</h4>
            <table>
                <tr><td>Fed Funds Rate</td><td style="text-align: right; font-weight: 700;">{fed_funds:.2f}%</td></tr>
                <tr><td>2Y Treasury</td><td style="text-align: right;">{treasury_2y:.2f}%</td></tr>
                <tr><td>10Y Treasury</td><td style="text-align: right;">{treasury_10y:.2f}%</td></tr>
                <tr><td>10Y-2Y Spread</td><td style="text-align: right;" class="{'text-red' if spread < 0 else 'text-green'}">{spread:.2f}%</td></tr>
            </table>
        </div>
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">유동성 지표</h4>
            <table>
                <tr><td>Net Liquidity</td><td style="text-align: right; font-weight: 700;">${fred.get('net_liquidity', 0):,.0f}B</td></tr>
                <tr><td>Fed Balance Sheet</td><td style="text-align: right;">${fred.get('fed_balance_sheet', 0):,.0f}B</td></tr>
                <tr><td>RRP</td><td style="text-align: right;">${fred.get('rrp', 0):,.0f}B</td></tr>
                <tr><td>TGA</td><td style="text-align: right;">${fred.get('tga', 0):,.0f}B</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 12px; font-size: 0.85rem;">
                ℹ️ 국제 시장 데이터는 실시간 수집 시 표시됩니다
            </p>
        </div>
    </div>
</div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🌍 국제 시장 분석</span>
    </div>
    <div class="grid grid-2" style="margin-bottom: 0;">
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">글로벌 지수</h4>
            <table>
                <tr><td>DAX (독일)</td><td style="text-align: right;">{dax_price}</td><td class="{dax_col}">{dax_chg}</td></tr>
                <tr><td>FTSE 100 (영국)</td><td style="text-align: right;">{ftse_price}</td><td class="{ftse_col}">{ftse_chg}</td></tr>
                <tr><td>Nikkei 225 (일본)</td><td style="text-align: right;">{nikkei_price}</td><td class="{nikkei_col}">{nikkei_chg}</td></tr>
                <tr><td>Shanghai (중국)</td><td style="text-align: right;">{shanghai_price}</td><td class="{shanghai_col}">{shanghai_chg}</td></tr>
                <tr><td>KOSPI (한국)</td><td style="text-align: right;">{kospi_price}</td><td class="{kospi_col}">{kospi_chg}</td></tr>
            </table>
        </div>
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">원자재</h4>
            <table>
                <tr><td>Gold</td><td style="text-align: right;">{gold_price}</td><td class="{gold_col}">{gold_chg}</td></tr>
                <tr><td>WTI 원유</td><td style="text-align: right;">{wti_price}</td><td class="{wti_col}">{wti_chg}</td></tr>
                <tr><td>Copper</td><td style="text-align: right;">{copper_price}</td><td class="{copper_col}">{copper_chg}</td></tr>
                <tr><td>DXY (달러)</td><td style="text-align: right;">{dxy_price}</td><td class="{dxy_col}">{dxy_chg}</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 12px; font-size: 0.85rem;">
                ⚠️ 안전자산 선호 및 원자재 시장 변동성 주시
            </p>
        </div>
    </div>
</div>'''

    def _generate_ark_invest_section(self) -> str:
        """ARK Invest 상세 분석"""
        data = self.integrated_data
        ark = data.get('ark_analysis', {})
        ai_raw = self.ai_report_raw if isinstance(self.ai_report_raw, dict) else {}

        if not ark:
            return ''

        # 상세 데이터
        top_increases = ark.get('top_increases', [])[:5]
        top_decreases = ark.get('top_decreases', [])[:5]
        signals = ark.get('signals', [])

        # ARK 데이터가 빈 경우, AI 리포트 주목 종목으로 보조 표시
        if not top_increases and isinstance(ai_raw.get('notable_stocks'), list):
            for stock in ai_raw.get('notable_stocks', [])[:5]:
                if not isinstance(stock, dict):
                    continue
                ticker = str(stock.get('ticker', '')).strip()
                chg = stock.get('change_1d', 0.0)
                if not ticker:
                    continue
                try:
                    chg_value = float(chg)
                except Exception:
                    chg_value = 0.0
                if chg_value >= 0:
                    top_increases.append({
                        'ticker': ticker,
                        'sector': 'AI Watchlist',
                        'weight_change_1d': chg_value,
                        'etf_count': 0,
                    })
                else:
                    top_decreases.append({
                        'ticker': ticker,
                        'sector': 'AI Watchlist',
                        'weight_change_1d': chg_value,
                        'etf_count': 0,
                    })
                if len(top_increases) >= 5 and len(top_decreases) >= 5:
                    break

        if not signals and isinstance(ai_raw.get('notable_stocks'), list):
            for stock in ai_raw.get('notable_stocks', [])[:3]:
                if not isinstance(stock, dict):
                    continue
                ticker = stock.get('ticker')
                reason = stock.get('notable_reason')
                if ticker and reason:
                    signals.append(f"{ticker}: {reason}")

        # 상세 테이블 생성
        inc_rows = ''
        for item in top_increases:
            ticker = item.get('ticker', 'N/A')
            sector = item.get('sector', '')
            weight_chg = item.get('weight_change_1d', 0)
            etf_count = item.get('etf_count', 0)
            inc_rows += f'''<tr>
                <td><strong>{ticker}</strong></td>
                <td>{sector}</td>
                <td class="text-green">+{weight_chg:.2f}%p</td>
                <td>{etf_count} ETF</td>
            </tr>'''

        dec_rows = ''
        for item in top_decreases:
            ticker = item.get('ticker', 'N/A')
            sector = item.get('sector', '')
            weight_chg = item.get('weight_change_1d', 0)
            etf_count = item.get('etf_count', 0)
            dec_rows += f'''<tr>
                <td><strong>{ticker}</strong></td>
                <td>{sector}</td>
                <td class="text-red">{weight_chg:.2f}%p</td>
                <td>{etf_count} ETF</td>
            </tr>'''

        signals_html = ''.join([f'<li>{s}</li>' for s in signals[:6]])

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🦋 ARK Invest 기관 수급</span>
    </div>
    <div class="grid grid-2">
        <div>
            <h4 class="text-green" style="margin-bottom: 12px;">비중 증가 (Top 5)</h4>
            <table>
                <tr><th>Ticker</th><th>섹터</th><th>변화</th><th>ETF</th></tr>
                {inc_rows if inc_rows else '<tr><td colspan="4">데이터 없음</td></tr>'}
            </table>
        </div>
        <div>
            <h4 class="text-red" style="margin-bottom: 12px;">비중 감소 (Top 5)</h4>
            <table>
                <tr><th>Ticker</th><th>섹터</th><th>변화</th><th>ETF</th></tr>
                {dec_rows if dec_rows else '<tr><td colspan="4">데이터 없음</td></tr>'}
            </table>
        </div>
    </div>
    <div style="margin-top: 16px; background: var(--bg-tertiary); padding: 16px; border-radius: 8px;">
        <h4 style="margin-bottom: 8px;">📌 주요 시그널</h4>
        <ul style="margin-left: 20px; font-size: 0.9rem;">
            {signals_html if signals_html else '<li>시그널 없음</li>'}
        </ul>
    </div>
</div>'''

    def _generate_market_structure_section(self) -> str:
        """시장 구조 분석 (DTW/DBSCAN)"""
        data = self.integrated_data
        dtw = data.get('dtw_similarity', {})
        dbscan = data.get('dbscan_outliers', {})

        # DTW 데이터
        most_similar = dtw.get('most_similar_pair', {})
        most_dissimilar = dtw.get('most_dissimilar_pair', {})
        lead_lag = dtw.get('lead_lag_spy_qqq', {})

        lead_asset = lead_lag.get('lead_asset', 'SPY')
        lag_asset = lead_lag.get('lag_asset', 'QQQ')
        optimal_lag = lead_lag.get('optimal_lag', 1)

        # DBSCAN 데이터
        outlier_ratio = dbscan.get('outlier_ratio', 0)
        n_outliers = dbscan.get('n_outliers', 0)
        outlier_tickers = dbscan.get('outlier_tickers', [])
        normal_tickers = dbscan.get('normal_tickers', [])

        if isinstance(outlier_ratio, float) and outlier_ratio <= 1:
            outlier_pct = outlier_ratio * 100
        else:
            outlier_pct = outlier_ratio

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🔬 시장 구조 분석 (DTW/DBSCAN)</span>
    </div>
    <div class="grid grid-2" style="margin-bottom: 0;">
        <div>
            <h4 style="margin-bottom: 12px; color: var(--accent-blue);">DTW 시계열 유사성</h4>
            <table>
                <tr>
                    <td>가장 유사한 쌍</td>
                    <td><strong>{most_similar.get('asset1', 'QQQ')} ↔ {most_similar.get('asset2', 'SPY')}</strong></td>
                </tr>
                <tr>
                    <td>가장 다른 쌍</td>
                    <td><strong>{most_dissimilar.get('asset1', 'VIX')} ↔ {most_dissimilar.get('asset2', 'UUP')}</strong></td>
                </tr>
                <tr>
                    <td>선행-후행 관계</td>
                    <td class="text-blue"><strong>{lead_asset}이(가) {lag_asset}보다 {optimal_lag}일 선행</strong></td>
                </tr>
            </table>
        </div>
        <div>
            <h4 style="margin-bottom: 12px; color: var(--accent-red);">DBSCAN 이상치 탐지</h4>
            <p style="font-size: 1.5rem; font-weight: 800; color: var(--accent-red);">{outlier_pct:.1f}%</p>
            <p class="text-muted">({n_outliers}개 자산이 이상치로 분류)</p>
            <p style="margin-top: 8px; font-size: 0.9rem;">
                <span class="text-red">이상치:</span> {', '.join(outlier_tickers[:5]) if outlier_tickers else 'N/A'}
            </p>
            <p style="font-size: 0.9rem;">
                <span class="text-green">정상:</span> {', '.join(normal_tickers) if normal_tickers else 'N/A'}
            </p>
        </div>
    </div>
</div>'''

    def _generate_volume_shock_section(self) -> str:
        """거래량 이상징후 및 충격 전파 그래프"""
        data = self.integrated_data

        # 거래량 이상징후
        vol_anomalies = data.get('volume_anomalies', [])

        # 충격 전파 그래프
        shock = data.get('shock_propagation', {})
        impact_score = shock.get('impact_score', 0)
        contagion_path = shock.get('contagion_path', [])
        vulnerable = shock.get('vulnerable_assets', [])
        details = shock.get('details', {})
        graph_nodes = details.get('graph_nodes', 0)
        paths_found = details.get('paths_found', 0)

        # 거래량 이상징후 HTML
        vol_html = ''
        if vol_anomalies:
            for va in vol_anomalies[:5]:
                ticker = va.get('ticker', 'N/A')
                severity = va.get('severity', 'LOW')
                desc = va.get('description', '') or '거래량 이상 감지'

                sev_color = '#c92a2a' if severity == 'HIGH' else '#f08c00' if severity == 'MEDIUM' else '#868e96'
                vol_html += f'''
                <div style="display: flex; align-items: center; gap: 12px; padding: 10px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 8px;">
                    <span style="font-weight: 700; width: 80px;">{ticker}</span>
                    <span style="background: {sev_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">{severity}</span>
                    <span style="flex: 1; font-size: 0.85rem; color: var(--text-secondary);">{desc[:50]}</span>
                </div>'''
        else:
            vol_html = '<p class="text-muted">탐지된 이상징후 없음</p>'

        # 충격 전파 경로 HTML
        path_html = ''
        if contagion_path:
            path_str = ' → '.join(contagion_path)
            path_html = f'<p style="font-family: monospace; background: var(--bg-tertiary); padding: 12px; border-radius: 6px;">{path_str}</p>'
        else:
            path_html = '<p class="text-muted">활성 전파 경로 없음 (시장 안정)</p>'

        # 취약 자산 HTML
        vuln_html = ''
        if vulnerable:
            vuln_html = f'<p><span class="text-red">취약 자산:</span> {", ".join(vulnerable[:5])}</p>'

        # 영향 점수 색상
        impact_color = '#c92a2a' if impact_score > 70 else '#f08c00' if impact_score > 30 else '#2b8a3e'

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-yellow);">
    <div class="card-header">
        <span class="card-title">📊 거래량 이상징후 & 충격 전파 그래프</span>
    </div>
    <div class="grid grid-2">
        <!-- 거래량 이상징후 -->
        <div>
            <h4 style="margin-bottom: 12px; color: var(--accent-yellow);">📈 거래량 이상징후 (Volume Anomalies)</h4>
            <p class="text-muted" style="font-size: 0.85rem; margin-bottom: 12px;">
                정상 범위를 벗어난 거래량 패턴 탐지 (Z-score 기반)
            </p>
            {vol_html}
        </div>

        <!-- 충격 전파 그래프 -->
        <div>
            <h4 style="margin-bottom: 12px; color: var(--accent-cyan);">🕸️ 충격 전파 그래프 (Shock Propagation)</h4>
            <div style="display: flex; gap: 20px; margin-bottom: 16px;">
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: {impact_color};">{impact_score:.0f}</p>
                    <p class="text-muted" style="font-size: 0.8rem;">영향 점수</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: var(--accent-blue);">{graph_nodes}</p>
                    <p class="text-muted" style="font-size: 0.8rem;">네트워크 노드</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: var(--accent-purple);">{paths_found}</p>
                    <p class="text-muted" style="font-size: 0.8rem;">전파 경로</p>
                </div>
            </div>
            <h5 style="margin-bottom: 8px;">전파 경로 (Contagion Path)</h5>
            {path_html}
            {vuln_html}
        </div>
    </div>
</div>'''

    def _generate_hft_microstructure(self) -> str:
        """HFT 미세구조 상세 (NEW)"""
        data = self.integrated_data
        hft = data.get('hft_microstructure', {})

        if not hft:
            return ''

        tick_rule = hft.get('tick_rule', {})
        buy_ratio = tick_rule.get('buy_ratio', 0.5)
        sell_ratio = tick_rule.get('sell_ratio', 0.5)
        tick_interp = tick_rule.get('interpretation', 'NEUTRAL')

        kyles = hft.get('kyles_lambda', {})
        lambda_val = kyles.get('lambda', 0)
        r_squared = kyles.get('r_squared', 0)
        kyle_interp = kyles.get('interpretation', 'N/A')

        vol_clock = hft.get('volume_clock', {})
        compression = vol_clock.get('compression_ratio', 0)

        # 색상 결정
        if buy_ratio > 0.55:
            tick_class = 'bg-green'
        elif buy_ratio < 0.45:
            tick_class = 'bg-red'
        else:
            tick_class = 'bg-yellow'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">⚡ HFT 시장 미세구조</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 0;">
        <div class="tech-item">
            <p class="tech-label">Tick Rule (매수/매도 비율)</p>
            <p class="tech-value">{buy_ratio*100:.1f}% / {sell_ratio*100:.1f}%</p>
            <span class="tech-badge {tick_class}">{tick_interp}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">Kyle's Lambda (가격 충격)</p>
            <p class="tech-value">{lambda_val:.2e}</p>
            <p class="text-muted" style="font-size: 0.85rem;">R² = {r_squared:.3f}</p>
            <span class="tech-badge bg-blue">{kyle_interp}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">Volume Clock 압축률</p>
            <p class="tech-value">{compression*100:.1f}%</p>
            <p class="text-muted" style="font-size: 0.85rem;">거래량 기반 시간 샘플링</p>
        </div>
    </div>
</div>'''

    def _generate_garch_volatility(self) -> str:
        """GARCH 변동성 상세 (NEW)"""
        data = self.integrated_data
        garch = data.get('garch_volatility', {})

        if not garch:
            return ''

        params = garch.get('garch_params', {})
        forecast = garch.get('volatility_forecast_10d', {})
        current_vol = garch.get('current_volatility', 0) * 100
        forecast_avg = garch.get('forecast_avg_volatility', 0) * 100

        persistence = params.get('persistence', 0)
        half_life = params.get('half_life', 0)

        # 10일 예측 리스트
        forecast_items = list(forecast.items())[:5]
        forecast_html = ' → '.join([f'D{k}: {v*100:.2f}%' for k, v in forecast_items])

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📉 GARCH 변동성 예측</span>
    </div>
    <div class="grid grid-2">
        <div>
            <div class="tech-item" style="margin-bottom: 16px;">
                <p class="tech-label">현재 변동성</p>
                <p class="tech-value text-red">{current_vol:.1f}%</p>
            </div>
            <div class="tech-item">
                <p class="tech-label">10일 평균 예측</p>
                <p class="tech-value text-blue">{forecast_avg:.1f}%</p>
                <span class="tech-badge {'bg-green' if forecast_avg < current_vol else 'bg-red'}">
                    {'감소 예상' if forecast_avg < current_vol else '증가 예상'}
                </span>
            </div>
        </div>
        <div>
            <p class="text-muted" style="margin-bottom: 8px;">GARCH 파라미터</p>
            <table>
                <tr><td>지속성 (Persistence)</td><td><strong>{persistence:.4f}</strong></td></tr>
                <tr><td>반감기 (Half-Life)</td><td><strong>{half_life:.1f}일</strong></td></tr>
            </table>
            <p style="margin-top: 12px; font-size: 0.85rem; color: var(--text-secondary);">
                {forecast_html}
            </p>
        </div>
    </div>
</div>'''

    def _generate_information_flow(self) -> str:
        """Information Flow (CAPM, 이상거래) (NEW)"""
        data = self.integrated_data
        info = data.get('information_flow', {})

        if not info:
            return ''

        # 이상 거래량
        abnormal = info.get('abnormal_volume', {})
        abnormal_days = abnormal.get('total_abnormal_days', 0)
        abnormal_ratio = abnormal.get('abnormal_ratio', 0)

        # CAPM
        capm_qqq = info.get('capm_QQQ', {})
        capm_tlt = info.get('capm_TLT', {})
        capm_gld = info.get('capm_GLD', {})

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📡 정보 흐름 분석</span>
    </div>
    <div class="grid grid-2">
        <div>
            <h4 style="margin-bottom: 12px;">이상 거래량 탐지</h4>
            <div class="tech-item">
                <p class="tech-value">{abnormal_days}일</p>
                <p class="text-muted">이상 거래일 ({abnormal_ratio*100:.1f}%)</p>
                <span class="tech-badge bg-green">안정적</span>
            </div>
        </div>
        <div>
            <h4 style="margin-bottom: 12px;">CAPM Alpha/Beta 분석</h4>
            <table>
                <tr>
                    <th>자산</th>
                    <th>Alpha (연율)</th>
                    <th>Beta</th>
                    <th>해석</th>
                </tr>
                <tr>
                    <td>QQQ</td>
                    <td class="text-green">{capm_qqq.get('alpha', 0)*252*100:.1f}%</td>
                    <td>{capm_qqq.get('beta', 1):.2f}</td>
                    <td>{capm_qqq.get('beta_interpretation', 'MARKET')[:20]}</td>
                </tr>
                <tr>
                    <td>TLT</td>
                    <td class="text-green">{capm_tlt.get('alpha', 0)*252*100:.1f}%</td>
                    <td>{capm_tlt.get('beta', 0):.2f}</td>
                    <td>{capm_tlt.get('beta_interpretation', 'INDEPENDENT')[:20]}</td>
                </tr>
                <tr>
                    <td>GLD</td>
                    <td class="text-green">{capm_gld.get('alpha', 0)*252*100:.1f}%</td>
                    <td>{capm_gld.get('beta', 0):.2f}</td>
                    <td>{capm_gld.get('beta_interpretation', 'INDEPENDENT')[:20]}</td>
                </tr>
            </table>
        </div>
    </div>
</div>'''

    def _generate_proof_of_index(self) -> str:
        """Proof-of-Index 상세 (NEW) - 파이 차트 포함"""
        data = self.integrated_data
        poi = data.get('proof_of_index', {})

        if not poi:
            return ''

        index_value = poi.get('index_value', 0)
        weights = poi.get('weights', {})
        verification = poi.get('verification', {})
        is_valid = verification.get('is_valid', True)
        hash_value = poi.get('hash', '')[:16] + '...' if poi.get('hash') else 'N/A'

        mean_rev = poi.get('mean_reversion_signal', {})
        z_score = mean_rev.get('z_score', 0)
        signal = mean_rev.get('signal', 'HOLD')

        # 파이 차트 생성 (conic-gradient)
        colors = ['#1864ab', '#5f3dc4', '#2b8a3e', '#f08c00', '#c92a2a', '#0b7285', '#868e96', '#e64980', '#7048e8', '#20c997']
        gradients = []
        legend_items = []
        cumulative = 0

        # 가중치 정렬 (큰 순서)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        total = sum(w for _, w in sorted_weights) if sorted_weights else 1

        for i, (ticker, weight) in enumerate(sorted_weights):
            pct = (weight / total * 100) if total > 0 else 0
            color = colors[i % len(colors)]
            gradients.append(f"{color} {cumulative}% {cumulative + pct}%")
            legend_items.append(f'''<div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                <div style="width: 12px; height: 12px; background: {color}; border-radius: 2px;"></div>
                <span style="font-size: 0.8rem;">{ticker}: {pct:.1f}%</span>
            </div>''')
            cumulative += pct

        # SVG 파이 차트 데이터 준비
        pie_data = [(ticker, weight / total * 100, colors[i % len(colors)])
                    for i, (ticker, weight) in enumerate(sorted_weights)]
        svg_chart = generate_svg_pie_chart(pie_data, size=160, hole_size=70, center_text="가중치")

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🔐 Proof-of-Index (블록체인 검증)</span>
        <span class="metric-badge {'bg-green' if is_valid else 'bg-red'}">{'✅ VERIFIED' if is_valid else '❌ FAILED'}</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 0;">
        <div>
            <div class="tech-item" style="margin-bottom: 16px;">
                <p class="tech-label">Index Value</p>
                <p class="tech-value">{index_value:.2f}</p>
            </div>
            <div class="tech-item">
                <p class="tech-label">Mean Reversion Z-Score</p>
                <p class="tech-value">{z_score:.2f}</p>
                <span class="tech-badge bg-blue">{signal}</span>
            </div>
            <p class="text-muted" style="margin-top: 12px; font-size: 0.75rem;">Hash: {hash_value}</p>
        </div>
        <div style="display: flex; justify-content: center; align-items: center;">
            {svg_chart}
        </div>
        <div>
            <p class="tech-label" style="margin-bottom: 12px;">Index 구성 가중치</p>
            {''.join(legend_items)}
        </div>
    </div>
</div>'''

    def _generate_debate_section(self) -> str:
        """멀티 에이전트 토론"""
        data = self.integrated_data
        full_pos = data.get('full_mode_position', 'NEUTRAL')
        ref_pos = data.get('reference_mode_position', 'NEUTRAL')
        modes_agree = data.get('modes_agree', True)

        def get_class(pos):
            p = pos.lower()
            if 'bull' in p:
                return 'bullish'
            elif 'bear' in p:
                return 'bearish'
            return 'neutral'

        consensus_text = '만장일치 (Consensus Reached)' if modes_agree else '의견 불일치 (Dissent)'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🤖 멀티 에이전트 토론</span>
    </div>
    <div class="grid grid-2" style="margin-bottom: 16px;">
        <div class="debate-box {get_class(full_pos)}">
            <div class="debate-title">
                <span>FULL Mode (365일 심층)</span>
                <span style="font-weight: 700;">{full_pos}</span>
            </div>
            <p class="debate-content">장기 데이터 기반 분석. 유동성 풍부, 기업 실적 견고.</p>
        </div>
        <div class="debate-box {get_class(ref_pos)}">
            <div class="debate-title">
                <span>REF Mode (90일 신속)</span>
                <span style="font-weight: 700;">{ref_pos}</span>
            </div>
            <p class="debate-content">단기 모멘텀 강세, 기술적 지표 골든 크로스.</p>
        </div>
    </div>
    <div class="consensus-box" style="background: var(--accent-{'green' if modes_agree else 'yellow'}-bg); border-color: var(--accent-{'green' if modes_agree else 'yellow'});">
        <strong>최종 합의: {consensus_text}</strong>
    </div>
</div>'''

    def _generate_institutional_narrative(self) -> str:
        """기관 투자자 분석 내러티브 (JP Morgan, Goldman, Berkshire) - NEW 2026-01-31"""
        data = self.integrated_data
        inst_analysis = data.get('institutional_analysis', {})

        if not inst_analysis:
            return ''

        narrative = inst_analysis.get('narrative', '')
        methods = inst_analysis.get('methodology_applied', [])
        jpmorgan = inst_analysis.get('jpmorgan_framework', {})
        gap_bridging = inst_analysis.get('gap_bridging', {})
        risk_quant = inst_analysis.get('risk_premium_quantification', {})

        methods_html = ''.join([f'<span class="metric-badge bg-purple">{m}</span> ' for m in methods[:4]])

        jpmorgan_html = ''
        if jpmorgan:
            stage = jpmorgan.get('consensus_position', 'N/A')
            conf = jpmorgan.get('confidence', 0.5)
            jpmorgan_html = f'''
            <div class="tech-item">
                <p class="tech-label">JP Morgan 5단계 버블 평가</p>
                <p class="tech-value text-purple">{stage[:40]}...</p>
                <p class="text-muted" style="font-size: 0.85rem;">신뢰도: {conf:.0%}</p>
            </div>'''

        gap_html = ''
        if gap_bridging:
            market_exp = gap_bridging.get('market_expectation', 'N/A')
            model_fc = gap_bridging.get('model_forecast', 'N/A')
            gap_status = gap_bridging.get('gap_status', 'UNKNOWN')
            gap_color = 'text-green' if gap_status == 'ALIGNED' else 'text-yellow'
            gap_html = f'''
            <div class="tech-item">
                <p class="tech-label">Goldman Sachs Gap-Bridging</p>
                <p class="tech-value {gap_color}">{gap_status}</p>
                <p class="text-muted" style="font-size: 0.85rem;">시장 기대: {market_exp} / 모델 예측: {model_fc}</p>
            </div>'''

        risk_html = ''
        if risk_quant:
            primary_risk = risk_quant.get('primary_risk_source', 'N/A')
            contribution = risk_quant.get('risk_contribution', 'N/A')
            risk_html = f'''
            <div class="tech-item">
                <p class="tech-label">Bekaert VIX 분해</p>
                <p class="tech-value">{primary_risk}</p>
                <p class="text-muted" style="font-size: 0.85rem;">기여도: {contribution}</p>
            </div>'''

        narrative_html = ''
        if narrative:
            narrative_html = f'''
            <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; margin-top: 16px;">
                <p style="font-style: italic; line-height: 1.8;">{narrative}</p>
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-cyan);">
    <div class="card-header">
        <span class="card-title">🏦 기관 투자자 관점 (Institutional View)</span>
    </div>
    <div style="margin-bottom: 12px;">
        <p class="text-muted" style="font-size: 0.85rem;">적용된 방법론:</p>
        {methods_html}
    </div>
    <div class="grid grid-3">
        {jpmorgan_html}
        {gap_html}
        {risk_html}
    </div>
    {narrative_html}
</div>'''

    def _generate_ai_institutional_interpretation(self) -> str:
        """AI 기관 분석 해석 (NEW) - Claude/GPT가 기관별 분석 결과를 종합 해석"""
        data = self.integrated_data

        # 데이터 수집
        bubble = data.get('bubble_framework', {})
        gap = data.get('gap_analysis', {})
        fomc = data.get('fomc_analysis', {})
        institutional = data.get('institutional_analysis', {})

        # 버블 프레임워크 해석
        bubble_stage = bubble.get('stage', 'UNKNOWN')
        bubble_score = bubble.get('total_score', 0)
        bubble_stages = bubble.get('stage_results', [])

        # Gap 분석 해석
        gap_signal = gap.get('overall_signal', 'NEUTRAL')
        gap_opportunity = gap.get('opportunity', '')
        gaps = gap.get('gaps', [])

        # FOMC 해석
        fomc_stance = fomc.get('stance', 'NEUTRAL')
        fomc_uncertainty = fomc.get('uncertainty', {}).get('policy_uncertainty_index', 50)
        fomc_interpretation = fomc.get('interpretation', '')

        # CSS 클래스 결정
        bubble_class = 'text-green' if bubble_score < 30 else 'text-yellow' if bubble_score < 60 else 'text-red'
        gap_class = 'text-green' if gap_signal == 'BULLISH' else 'text-red' if gap_signal == 'BEARISH' else 'text-yellow'
        fomc_class = 'text-red' if fomc_stance == 'HAWKISH' else 'text-green' if fomc_stance == 'DOVISH' else 'text-yellow'

        # 버블 단계 시각화 바
        bubble_stages_html = ''
        for stage in bubble_stages:
            stage_name = stage.get('stage', '').replace('_', ' ').title()
            stage_passed = stage.get('passed', False)
            stage_score = stage.get('score', 0)
            evidence = stage.get('evidence', '')[:60]
            icon = '✅' if stage_passed else '❌'
            bubble_stages_html += f'''
            <div style="display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background: var(--bg-tertiary); border-radius: 6px;">
                <span style="width: 24px;">{icon}</span>
                <span style="flex: 1; font-weight: 500;">{stage_name}</span>
                <span style="width: 60px; text-align: right; font-weight: 600;">{stage_score:.1f}</span>
            </div>'''

        # Gap 분석 시각화
        gap_items_html = ''
        for g in gaps[:4]:
            metric = g.get('metric', '').replace('_', ' ').title()
            market_val = g.get('market_implied', 0)
            model_val = g.get('model_forecast', 0)
            gap_val = g.get('gap', 0)
            signal = g.get('signal', 'NEUTRAL')
            signal_class = 'text-green' if signal == 'BULLISH' else 'text-red' if signal == 'BEARISH' else 'text-yellow'

            # 바 차트 (시장 vs 모델)
            max_val = max(abs(market_val), abs(model_val), 0.01)
            market_pct = min((market_val / max_val) * 100, 100)
            model_pct = min((model_val / max_val) * 100, 100)

            gap_items_html += f'''
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 500;">{metric}</span>
                    <span class="{signal_class}" style="font-weight: 700;">{signal}</span>
                </div>
                <div style="display: flex; gap: 4px; height: 20px;">
                    <div style="background: var(--accent-blue); width: {market_pct:.0f}%; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted);">
                    <span>시장: {market_val:.2f}</span>
                    <span>모델: {model_val:.2f}</span>
                    <span>Gap: {gap_val:+.2f}</span>
                </div>
            </div>'''

        # FOMC 분포 시각화
        member_dist = fomc.get('member_distribution', {})
        hawkish_count = member_dist.get('hawkish', 0)
        neutral_count = member_dist.get('neutral', 0)
        dovish_count = member_dist.get('dovish', 0)
        total = member_dist.get('total', 1) or 1

        hawkish_pct = (hawkish_count / total) * 100
        neutral_pct = (neutral_count / total) * 100
        dovish_pct = (dovish_count / total) * 100

        # AI 종합 해석 생성 (간단한 규칙 기반)
        ai_interpretation = self._generate_ai_synthesis(
            bubble_score=bubble_score,
            bubble_stage=bubble_stage,
            gap_signal=gap_signal,
            gap_opportunity=gap_opportunity,
            fomc_stance=fomc_stance,
            fomc_uncertainty=fomc_uncertainty
        )

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-purple);">
    <div class="card-header">
        <span class="card-title">🧠 AI 기관 분석 해석</span>
        <span class="text-muted" style="font-size: 0.85rem;">Claude + GPT Multi-LLM Synthesis</span>
    </div>

    <!-- 3열 그리드: 버블/Gap/FOMC -->
    <div class="grid grid-3" style="margin-bottom: 20px;">
        <!-- 버블 프레임워크 -->
        <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px;">
            <h4 style="margin-bottom: 12px; color: var(--text-primary);">📊 5-Stage Bubble Framework</h4>
            <div style="text-align: center; margin-bottom: 12px;">
                <span class="{bubble_class}" style="font-size: 2rem; font-weight: 700;">{bubble_score:.0f}</span>
                <span style="font-size: 0.9rem; color: var(--text-muted);">/100</span>
            </div>
            <div style="text-align: center; margin-bottom: 12px;">
                <span class="signal-badge {'bullish' if bubble_score < 40 else 'bearish' if bubble_score > 70 else 'neutral'}">{bubble_stage}</span>
            </div>
            {bubble_stages_html}
        </div>

        <!-- Gap Analysis -->
        <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px;">
            <h4 style="margin-bottom: 12px; color: var(--text-primary);">📈 Market-Model Gap</h4>
            <div style="text-align: center; margin-bottom: 12px;">
                <span class="signal-badge {'bullish' if gap_signal == 'BULLISH' else 'bearish' if gap_signal == 'BEARISH' else 'neutral'}">{gap_signal}</span>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px;">{gap_opportunity}</p>
            {gap_items_html}
        </div>

        <!-- FOMC Analysis -->
        <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px;">
            <h4 style="margin-bottom: 12px; color: var(--text-primary);">🏛️ FOMC Dot Plot</h4>
            <div style="text-align: center; margin-bottom: 12px;">
                <span class="signal-badge {'bearish' if fomc_stance == 'HAWKISH' else 'bullish' if fomc_stance == 'DOVISH' else 'neutral'}">{fomc_stance}</span>
            </div>
            <div style="margin-bottom: 16px;">
                <div style="display: flex; height: 24px; border-radius: 6px; overflow: hidden;">
                    <div style="background: #c92a2a; width: {hawkish_pct:.0f}%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.75rem;">
                        {hawkish_count}
                    </div>
                    <div style="background: #868e96; width: {neutral_pct:.0f}%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.75rem;">
                        {neutral_count}
                    </div>
                    <div style="background: #2b8a3e; width: {dovish_pct:.0f}%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.75rem;">
                        {dovish_count}
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">
                    <span>Hawkish</span>
                    <span>Neutral</span>
                    <span>Dovish</span>
                </div>
            </div>
            <div style="background: var(--bg-secondary); padding: 12px; border-radius: 6px;">
                <p style="font-size: 0.85rem; margin-bottom: 8px;">정책 불확실성 지수</p>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; background: var(--border); height: 8px; border-radius: 4px;">
                        <div style="background: {'#c92a2a' if fomc_uncertainty > 70 else '#f08c00' if fomc_uncertainty > 40 else '#2b8a3e'}; width: {fomc_uncertainty:.0f}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <span style="font-weight: 600; width: 40px;">{fomc_uncertainty:.0f}</span>
                </div>
            </div>
        </div>
    </div>

    <!-- AI 종합 해석 -->
    <div style="background: linear-gradient(135deg, var(--accent-purple-bg), var(--accent-blue-bg)); padding: 20px; border-radius: 10px; border: 1px solid var(--accent-purple);">
        <h4 style="margin-bottom: 12px; color: var(--accent-purple);">🤖 AI 종합 해석 (Multi-LLM Consensus)</h4>
        <p style="line-height: 1.8; color: var(--text-primary);">{ai_interpretation}</p>
    </div>
</div>'''

    def _generate_ai_synthesis(self, bubble_score: float, bubble_stage: str,
                                gap_signal: str, gap_opportunity: str,
                                fomc_stance: str, fomc_uncertainty: float) -> str:
        """AI 기관 분석 종합 해석 생성 (HTML 포맷)"""
        interpretations = []

        # 버블 해석
        if bubble_score < 30:
            interpretations.append(f"버블 프레임워크 점수 {bubble_score:.0f}점으로 <strong>안전 구간</strong>입니다. 현재 시장에 과열 징후는 관찰되지 않습니다.")
        elif bubble_score < 60:
            interpretations.append(f"버블 위험 점수 {bubble_score:.0f}점({bubble_stage})으로 <strong>초기 형성 단계</strong>입니다. 주의 깊은 모니터링이 필요하나 즉각적 리스크는 제한적입니다.")
        else:
            interpretations.append(f"버블 위험 점수 {bubble_score:.0f}점으로 <strong>경고 수준</strong>입니다. 포지션 축소 및 방어적 전략을 고려해야 합니다.")

        # Gap 해석
        if gap_signal == 'BULLISH':
            interpretations.append("시장-모델 갭 분석에서 시장이 과도하게 비관적이어서 <strong>매수 기회</strong>가 존재합니다.")
        elif gap_signal == 'BEARISH':
            interpretations.append(f"시장-모델 갭 분석에서 시장이 과도하게 낙관적입니다. {gap_opportunity}")
        else:
            interpretations.append("시장 내재 기대와 모델 예측이 대체로 일치하여 현재 <strong>균형 상태</strong>입니다.")

        # FOMC 해석
        if fomc_stance == 'HAWKISH':
            interpretations.append(f"FOMC 위원들이 긴축적 성향(불확실성 {fomc_uncertainty:.0f})을 보여 <strong>금리 인하 기대는 제한적</strong>입니다. 성장주보다 가치주, 배당주가 유리합니다.")
        elif fomc_stance == 'DOVISH':
            interpretations.append(f"FOMC가 완화적 성향을 보여 <strong>금리 인하 가능성</strong>이 높습니다. 성장주 및 기술주에 우호적입니다.")
        else:
            interpretations.append("FOMC의 정책 방향이 중립적이어서 당분간 현 금리 수준이 유지될 것으로 예상됩니다.")

        # 종합 권고
        bullish_signals = sum([
            bubble_score < 40,
            gap_signal == 'BULLISH',
            fomc_stance == 'DOVISH'
        ])
        bearish_signals = sum([
            bubble_score > 60,
            gap_signal == 'BEARISH',
            fomc_stance == 'HAWKISH'
        ])

        if bullish_signals >= 2:
            interpretations.append("<br><br><strong style='color: var(--accent-green);'>종합 판단: 강세 (BULLISH)</strong> - 복수의 기관 프레임워크가 긍정적 시그널을 보내고 있습니다. 리스크 자산 비중 확대를 고려하십시오.")
        elif bearish_signals >= 2:
            interpretations.append("<br><br><strong style='color: var(--accent-red);'>종합 판단: 약세 (BEARISH)</strong> - 복수의 기관 프레임워크가 경고 시그널을 보내고 있습니다. 방어적 포지셔닝을 권고합니다.")
        else:
            interpretations.append("<br><br><strong style='color: var(--accent-yellow);'>종합 판단: 중립 (NEUTRAL)</strong> - 혼재된 시그널로 인해 적극적 포지션 변경보다는 현 수준 유지가 적절합니다.")

        return ' '.join(interpretations)

    def _generate_school_interpretations(self) -> str:
        """학파별 해석 (NEW)"""
        data = self.integrated_data
        debate = data.get('debate_consensus', {})
        enhanced = debate.get('enhanced', {})
        interp = enhanced.get('interpretation', {})

        schools = interp.get('school_interpretations', [])
        if not schools:
            return ''

        school_html = ''
        for school in schools:
            name = school.get('school', 'Unknown')
            stance = school.get('stance', 'NEUTRAL')
            reasoning = school.get('reasoning', [])

            stance_class = 'text-green' if 'BULL' in stance else 'text-red' if 'BEAR' in stance else 'text-yellow'

            reasons_html = ''.join([f'<li>{r}</li>' for r in reasoning[:3]])

            school_html += f'''
            <div class="debate-box {'bullish' if 'BULL' in stance else 'bearish' if 'BEAR' in stance else 'neutral'}">
                <div class="debate-title">
                    <span>{name}</span>
                    <span class="{stance_class}" style="font-weight: 700;">{stance}</span>
                </div>
                <ul style="margin-left: 16px; font-size: 0.9rem;">
                    {reasons_html}
                </ul>
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🎓 경제학파별 해석 (Multi-LLM)</span>
    </div>
    {school_html}
</div>'''

    def _generate_reasoning_chain(self) -> str:
        """추론 과정 (Audit Trail) (NEW)"""
        data = self.integrated_data
        chain = data.get('reasoning_chain', [])

        if not chain:
            return ''

        steps_html = ''
        for step in chain:
            step_num = step.get('step', 0)
            agent = step.get('agent', 'Unknown')
            output = step.get('output', '')
            confidence = step.get('confidence', 0)
            factors = step.get('key_factors', [])

            factors_html = ', '.join(factors[:3]) if factors else 'N/A'

            steps_html += f'''
            <div class="reasoning-step">
                <div class="step-number">{step_num}</div>
                <div class="step-content">
                    <p class="step-agent">{agent}</p>
                    <p class="step-output">{output}</p>
                    <p class="step-confidence">신뢰도: {confidence:.0f}% | Key Factors: {factors_html}</p>
                </div>
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🔗 추론 과정 (Reasoning Chain)</span>
    </div>
    {steps_html}
</div>'''

    def _generate_portfolio_section(self) -> str:
        """추천 포트폴리오"""
        data = self.integrated_data
        weights = data.get('portfolio_weights', {})

        if not weights:
            # 현재 레짐 기반 기본 포트폴리오 생성
            recommendation = data.get('final_recommendation', 'NEUTRAL')
            risk_score = data.get('risk_score', 50)

            if 'BULL' in recommendation.upper() and risk_score < 40:
                weights = {'주식 (성장)': 45, '주식 (가치)': 25, '채권': 15, '원자재': 10, '현금': 5}
            elif 'BEAR' in recommendation.upper() or risk_score > 60:
                weights = {'채권': 35, '현금': 25, '주식 (방어)': 20, '금/원자재': 15, '인버스': 5}
            else:
                weights = {'주식 (균형)': 35, '채권': 25, '현금': 20, '원자재': 10, '대안투자': 10}

        colors = ['#1864ab', '#5f3dc4', '#2b8a3e', '#f08c00', '#868e96', '#c92a2a', '#0b7285']
        legend_items = []

        items = list(weights.items())[:7]
        total = sum(v for _, v in items)

        # SVG 파이 차트 데이터 준비
        pie_data = []
        for i, (label, value) in enumerate(items):
            pct = (value / total * 100) if total > 0 else 0
            color = colors[i % len(colors)]
            pie_data.append((label, pct, color))
            legend_items.append(f'''<div class="legend-item">
                <div class="legend-color" style="background: {color};"></div>
                <span>{label} ({pct:.0f}%)</span>
            </div>''')

        svg_chart = generate_svg_pie_chart(pie_data, size=180, hole_size=80, center_text="배분")

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">💼 추천 포트폴리오</span>
    </div>
    <div class="pie-container">
        <div style="display: flex; justify-content: center; align-items: center;">
            {svg_chart}
        </div>
        <div class="pie-legend">
            {''.join(legend_items)}
        </div>
    </div>
</div>'''

    def _generate_adaptive_portfolios(self) -> str:
        """적응형 포트폴리오 시그널 (NEW)"""
        data = self.integrated_data
        adaptive = data.get('adaptive_portfolios', {})

        if not adaptive:
            return ''

        aggressive = adaptive.get('aggressive', 'N/A')
        balanced = adaptive.get('balanced', 'N/A')
        conservative = adaptive.get('conservative', 'N/A')

        def get_signal_class(signal):
            if 'AGGRESSIVE' in signal or 'ENTRY' in signal:
                return 'bg-green', 'text-green'
            elif 'EXIT' in signal or 'DEFENSIVE' in signal:
                return 'bg-red', 'text-red'
            return 'bg-yellow', 'text-yellow'

        agg_bg, agg_txt = get_signal_class(aggressive)
        bal_bg, bal_txt = get_signal_class(balanced)
        con_bg, con_txt = get_signal_class(conservative)

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🎯 적응형 포트폴리오 시그널</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 0;">
        <div class="tech-item">
            <p class="tech-label">공격형 (Aggressive)</p>
            <p class="tech-value {agg_txt}">{aggressive}</p>
        </div>
        <div class="tech-item">
            <p class="tech-label">균형형 (Balanced)</p>
            <p class="tech-value {bal_txt}">{balanced}</p>
        </div>
        <div class="tech-item">
            <p class="tech-label">보수형 (Conservative)</p>
            <p class="tech-value {con_txt}">{conservative}</p>
        </div>
    </div>
</div>'''

    def _generate_sector_analysis(self) -> str:
        """섹터 분석 (NEW)"""
        # MD 섹션 12에서 추출
        section = self.ai_report_sections.get('section_12', {})

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📊 섹터별 투자 의견</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 0;">
        <div>
            <h4 class="text-green" style="margin-bottom: 12px;">강세 (Overweight)</h4>
            <table>
                <tr><td><strong>기술</strong></td><td>XLK</td><td class="text-green">AI, 반도체</td></tr>
                <tr><td><strong>소재</strong></td><td>XLB</td><td class="text-green">원자재 강세</td></tr>
            </table>
        </div>
        <div>
            <h4 class="text-yellow" style="margin-bottom: 12px;">중립 (Neutral)</h4>
            <table>
                <tr><td><strong>헬스케어</strong></td><td>XLV</td><td>장기 성장</td></tr>
                <tr><td><strong>금융</strong></td><td>XLF</td><td>금리 동결</td></tr>
            </table>
        </div>
        <div>
            <h4 class="text-red" style="margin-bottom: 12px;">약세 (Underweight)</h4>
            <table>
                <tr><td><strong>부동산</strong></td><td>XLRE</td><td class="text-red">금리 부담</td></tr>
                <tr><td><strong>중소형주</strong></td><td>IWM</td><td class="text-red">대형주 쏠림</td></tr>
            </table>
        </div>
    </div>
</div>'''

    def _generate_entry_exit_section(self) -> str:
        """진입/청산 전략"""
        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🎯 실행 전략 (SPY 기준)</span>
    </div>
    <div class="grid grid-2">
        <div>
            <h4 class="text-green" style="margin-bottom: 12px;">📥 진입 전략 (분할 매수)</h4>
            <table>
                <tr><th>단계</th><th>가격</th><th>비중</th><th>조건</th></tr>
                <tr><td>1차</td><td>현재가</td><td>30%</td><td>즉시 진입</td></tr>
                <tr><td>2차</td><td>-1.5%</td><td>30%</td><td>조정 시 매수</td></tr>
                <tr><td>3차</td><td class="text-green">-2.5%</td><td>40%</td><td>강력 지지선</td></tr>
            </table>
        </div>
        <div>
            <h4 class="text-red" style="margin-bottom: 12px;">📤 청산 전략 (목표가)</h4>
            <table>
                <tr><th>단계</th><th>목표가</th><th>비중</th><th>수익</th></tr>
                <tr><td>1차</td><td>저항선</td><td>50%</td><td>+2%</td></tr>
                <tr><td>2차</td><td>+3%</td><td>30%</td><td>+3~4%</td></tr>
                <tr><td class="text-red">손절</td><td class="text-red">-5%</td><td>전량</td><td>-5%</td></tr>
            </table>
        </div>
    </div>
</div>'''

    def _extract_watchlist_items(self, content: str) -> list:
        """MD 주식 목록 파싱"""
        import re
        items = []
        # Split by level 3 header (### Ticker)
        parts = re.split(r'^### ', content, flags=re.MULTILINE)
        
        for part in parts:
            part = part.strip()
            if not part or part.startswith('#'): continue
            
            lines = part.splitlines()
            ticker = lines[0].strip()
            
            # Basic data
            item = {'ticker': ticker, '1d': 'N/A', '5d': 'N/A', '20d': 'N/A', 'reason': ''}
            
            # Join rest of lines for searching
            body = '\n'.join(lines[1:])
            
            # Extract metrics
            d1 = re.search(r'- 1일 변화: (.*?)$', body, re.MULTILINE)
            d5 = re.search(r'- 5일 변화: (.*?)$', body, re.MULTILINE)
            d20 = re.search(r'- 20일 변화: (.*?)$', body, re.MULTILINE)
            reason = re.search(r'- \*\*주목 이유\*\*: (.*?)$', body, re.MULTILINE)
            
            if d1: item['1d'] = d1.group(1).strip()
            if d5: item['5d'] = d5.group(1).strip()
            if d20: item['20d'] = d20.group(1).strip()
            if reason: item['reason'] = reason.group(1).strip()
            
            items.append(item)
            
        return items

    def _generate_watchlist_section(self) -> str:
        """주목할 종목 (NEW) - ARK 데이터 기반으로 생성"""
        data = self.integrated_data
        ai_raw = self.ai_report_raw if isinstance(self.ai_report_raw, dict) else {}

        # MD 섹션 7에서 추출 시도
        section = self.ai_report_sections.get('section_7', {})
        content = section.get('content', '')

        items = self._extract_watchlist_items(content)
        no_data_reason = ""

        # 1차 fallback: AI raw report의 notable_stocks 사용
        if not items:
            raw_notable = ai_raw.get('notable_stocks', [])
            if isinstance(raw_notable, list):
                for stock in raw_notable[:6]:
                    if not isinstance(stock, dict):
                        continue
                    ticker = str(stock.get('ticker', '')).strip()
                    if not ticker:
                        continue
                    d1 = stock.get('change_1d')
                    d5 = stock.get('change_5d')
                    d20 = stock.get('change_20d')

                    items.append({
                        'ticker': ticker,
                        '1d': f"{float(d1):+.2f}%" if isinstance(d1, (int, float)) else 'N/A',
                        '5d': f"{float(d5):+.2f}%" if isinstance(d5, (int, float)) else 'N/A',
                        '20d': f"{float(d20):+.2f}%" if isinstance(d20, (int, float)) else 'N/A',
                        'reason': str(
                            stock.get('notable_reason')
                            or stock.get('news_summary')
                            or 'AI Report notable stock'
                        ).strip(),
                    })

            no_data_reason = str(ai_raw.get('notable_stocks_reason', '')).strip()

        # 2차 fallback: ARK 데이터 사용
        if not items:
            ark = data.get('ark_analysis', {})
            top_increases = ark.get('top_increases', [])[:3]
            consensus_buys = ark.get('consensus_buys', [])[:3]

            # ARK 데이터로 watchlist 생성
            for item in top_increases:
                ticker = item.get('ticker', '')
                if ticker:
                    items.append({
                        'ticker': ticker,
                        '1d': f"+{item.get('weight_change_1d', 0):.2f}%p",
                        '5d': 'N/A',
                        '20d': 'N/A',
                        'reason': f"ARK 비중 증가 ({item.get('etf_count', 0)} ETF)"
                    })

            for ticker in consensus_buys:
                if ticker and ticker not in [i['ticker'] for i in items]:
                    items.append({
                        'ticker': ticker,
                        '1d': 'N/A',
                        '5d': 'N/A',
                        '20d': 'N/A',
                        'reason': 'ARK Consensus Buy'
                    })

        html_cards = ""

        if not items:
            if no_data_reason:
                html_cards = f"<p class='text-muted'>{no_data_reason}</p>"
            else:
                html_cards = "<p class='text-muted'>현재 주목할 종목 데이터가 없습니다. ARK 분석 섹션을 참고하세요.</p>"

        for item in items[:6]:
            ticker = item['ticker']
            d1 = item.get('1d', 'N/A')
            d5 = item.get('5d', 'N/A')
            d20 = item.get('20d', 'N/A')
            reason = item.get('reason', '')

            # Determine badge/color
            badge_text = "중립"
            badge_class = "bg-blue"

            try:
                d1_val = float(str(d1).replace('%', '').replace('+', '').replace('p', ''))
                if d1_val > 2:
                    badge_text = "강세"
                    badge_class = "bg-green"
                elif d1_val < -2:
                    badge_text = "약세"
                    badge_class = "bg-red"
            except:
                pass

            # Formatting helpers
            def fmt_cls(val_str):
                if '-' in str(val_str): return 'text-red'
                if '+' in str(val_str): return 'text-green'
                return ''

            html_cards += f'''
        <div class="signal-card">
            <div class="signal-header">
                <span class="signal-ticker">{ticker}</span>
                <span class="signal-badge {badge_class}">{badge_text}</span>
            </div>
            <table style="font-size: 0.9rem;">
                <tr><td>1D</td><td class="{fmt_cls(d1)}">{d1}</td></tr>
                <tr><td>5D</td><td class="{fmt_cls(d5)}">{d5}</td></tr>
                <tr><td>20D</td><td class="{fmt_cls(d20)}">{d20}</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 8px; font-size: 0.85rem;">{reason}</p>
        </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">👀 주목할 종목</span>
    </div>
    <div class="grid grid-3">
        {html_cards}
    </div>
</div>'''

    def _generate_news_section(self) -> str:
        """주요 시장 뉴스 - 실시간 데이터 사용"""
        news_items = []
        ai_raw = self.ai_report_raw if isinstance(self.ai_report_raw, dict) else {}

        # 1. Perplexity 뉴스 (ai_report.section_8)
        section = self.ai_report_sections.get('section_8', {})
        perplexity_content = section.get('content', '')
        if (not perplexity_content or len(perplexity_content) < 50) and ai_raw.get('perplexity_news'):
            perplexity_content = str(ai_raw.get('perplexity_news', ''))

        if perplexity_content and len(perplexity_content) > 50:
            # Perplexity 응답을 뉴스 항목으로 파싱
            lines = perplexity_content.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith('#'):
                    # 태그 추론
                    tag, tag_class = self._infer_news_tag(line)
                    news_items.append({
                        'tag': tag,
                        'tag_class': tag_class,
                        'title': line[:80] + ('...' if len(line) > 80 else ''),
                        'content': line[80:160] if len(line) > 80 else ''
                    })
                    if len(news_items) >= 5:
                        break

        # 1.5 references 기반 보강
        if len(news_items) < 3:
            refs = ai_raw.get('references', [])
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, str):
                        continue
                    line = ref.strip()
                    if not line:
                        continue
                    tag, tag_class = self._infer_news_tag(line)
                    news_items.append({
                        'tag': tag,
                        'tag_class': tag_class,
                        'title': line[:80] + ('...' if len(line) > 80 else ''),
                        'content': 'AI Report reference'
                    })
                    if len(news_items) >= 5:
                        break

        # 2. yfinance 뉴스 실시간 수집 (Perplexity 없으면)
        if len(news_items) < 3:
            try:
                import yfinance as yf
                from dateutil import parser as date_parser

                spy = yf.Ticker('SPY')
                yf_news = spy.news[:5] if spy.news else []

                for item in yf_news:
                    content = item.get('content', {})
                    title = content.get('title', '') if content else item.get('title', '')
                    summary = content.get('summary', '')[:100] if content else ''

                    if title:
                        tag, tag_class = self._infer_news_tag(title)
                        news_items.append({
                            'tag': tag,
                            'tag_class': tag_class,
                            'title': title[:80] + ('...' if len(title) > 80 else ''),
                            'content': summary
                        })
                        if len(news_items) >= 5:
                            break
            except Exception as e:
                pass

        # 3. CNBC RSS (extended_data.news_sentiment)
        if len(news_items) < 3:
            ext = self.integrated_data.get('extended_data', {})
            news_sent = ext.get('news_sentiment', {})
            headline = news_sent.get('top_headline', '')
            if headline:
                tag, tag_class = self._infer_news_tag(headline)
                news_items.append({
                    'tag': tag,
                    'tag_class': tag_class,
                    'title': headline[:80] + ('...' if len(headline) > 80 else ''),
                    'content': f"Sentiment: {news_sent.get('label', 'Neutral')}"
                })

        # 4. 폴백: 기본 뉴스 (데이터 없을 때)
        if not news_items:
            news_items = [
                {'tag': 'Market', 'tag_class': 'bg-blue', 'title': '실시간 뉴스 수집 중...', 'content': 'Perplexity/yfinance API 연동 확인 필요'}
            ]

        news_html = ''
        for item in news_items[:5]:
            news_html += f'''<div class="news-card">
                <span class="news-tag {item['tag_class']}">{item['tag']}</span>
                <p class="news-title">{item['title']}</p>
                <p class="news-content">{item['content']}</p>
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📰 주요 시장 뉴스 (실시간)</span>
    </div>
    {news_html}
</div>'''

    def _infer_news_tag(self, text: str) -> tuple:
        """뉴스 텍스트에서 태그 추론"""
        text_lower = text.lower()
        if any(w in text_lower for w in ['fed', 'fomc', 'rate', 'powell', '금리', '연준']):
            return 'Fed', 'bg-purple'
        elif any(w in text_lower for w in ['tech', 'ai', 'nvidia', 'apple', 'microsoft', '기술']):
            return 'Tech', 'bg-green'
        elif any(w in text_lower for w in ['crypto', 'bitcoin', 'btc', 'eth', '비트코인']):
            return 'Crypto', 'bg-yellow'
        elif any(w in text_lower for w in ['oil', 'gold', 'commodity', '원유', '금']):
            return 'Commodity', 'bg-orange'
        elif any(w in text_lower for w in ['china', 'trade', 'tariff', '중국', '관세']):
            return 'Trade', 'bg-red'
        else:
            return 'Market', 'bg-blue'

    def _generate_scenario_section(self) -> str:
        """시나리오 분석 - 현재 데이터 기반 동적 생성"""
        data = self.integrated_data

        # 현재 데이터에서 시나리오 확률 추출
        regime = data.get('regime', {})
        regime_type = regime.get('regime', 'Neutral') if isinstance(regime, dict) else str(regime)
        risk_score = data.get('risk_score', 50)
        recommendation = data.get('final_recommendation', 'NEUTRAL')

        # AI 리포트에서 시나리오 정보 추출 시도
        scenarios = data.get('scenarios', {})
        ai_report = data.get('ai_report', {})
        if isinstance(ai_report, dict):
            scenarios = ai_report.get('scenarios', scenarios)

        # 시나리오 확률 계산 (현재 레짐 기반)
        if 'BULL' in regime_type.upper() or 'BULL' in recommendation.upper():
            base_prob, bull_prob, bear_prob = 50, 35, 15
        elif 'BEAR' in regime_type.upper() or 'BEAR' in recommendation.upper():
            base_prob, bull_prob, bear_prob = 45, 15, 40
        else:
            base_prob, bull_prob, bear_prob = 55, 25, 20

        # 리스크 점수에 따른 조정
        if risk_score > 60:
            bear_prob += 10
            bull_prob -= 5
            base_prob -= 5
        elif risk_score < 30:
            bull_prob += 10
            bear_prob -= 5
            base_prob -= 5

        # 시나리오별 설명 (현재 데이터 반영)
        warnings = data.get('warnings', [])
        events = data.get('events_detected', [])

        # 주요 위험 요소 추출
        risk_factors = []
        for w in warnings[:2]:
            if isinstance(w, str):
                risk_factors.append(w[:40])
            elif isinstance(w, dict):
                risk_factors.append(w.get('message', '')[:40])

        # 긍정 요소 추출
        positive_factors = []
        liquidity = data.get('fred_summary', {}).get('liquidity_regime', '')
        if 'abundant' in str(liquidity).lower():
            positive_factors.append('풍부한 유동성')
        if risk_score < 40:
            positive_factors.append('낮은 리스크 환경')

        # 동적 설명 생성
        base_desc = f"현재 {regime_type} 레짐 유지, 경제 지표 모니터링"
        bull_desc = ', '.join(positive_factors[:2]) if positive_factors else "경기 회복 가속화 시"
        bear_desc = ', '.join(risk_factors[:2]) if risk_factors else "리스크 요인 확대 시"

        return f'''
<div class="grid grid-3" style="margin-bottom: 24px;">
    <div class="scenario-card base">
        <div class="scenario-header">
            <span class="scenario-title text-blue">📊 Base Case</span>
            <span class="scenario-prob text-blue">{base_prob}%</span>
        </div>
        <p style="margin-bottom: 8px;">{base_desc}</p>
        <p style="font-weight: 700;">현재 추세 유지</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 현재 포지션 유지, 조정 시 매수</p>
    </div>
    <div class="scenario-card bull">
        <div class="scenario-header">
            <span class="scenario-title text-green">🐂 Bull Case</span>
            <span class="scenario-prob text-green">{bull_prob}%</span>
        </div>
        <p style="margin-bottom: 8px;">{bull_desc}</p>
        <p style="font-weight: 700;">상승 모멘텀 강화</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 주식 비중 확대, 성장주 집중</p>
    </div>
    <div class="scenario-card bear">
        <div class="scenario-header">
            <span class="scenario-title text-red">🐻 Bear Case</span>
            <span class="scenario-prob text-red">{bear_prob}%</span>
        </div>
        <p style="margin-bottom: 8px;">{bear_desc}</p>
        <p style="font-weight: 700;">하락 리스크 증가</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 현금/채권 확대, 방어적 포지션</p>
    </div>
</div>'''

    def _generate_final_proposal(self) -> str:
        """최종 제안 (NEW)"""
        data = self.integrated_data
        recommendation = data.get('final_recommendation', 'BULLISH')
        confidence = data.get('confidence', 0.7)
        if confidence <= 1:
            confidence *= 100

        # 액션 아이템
        if 'BULL' in recommendation.upper():
            action_items = ['주식 비중 확대 고려', '성장주/소형주 비중 점검', '레버리지 ETF 검토 가능']
            rec_class = 'bg-green'
            rec_text = '📈 적극적 매수'
        elif 'BEAR' in recommendation.upper():
            action_items = ['주식 비중 축소', '현금/채권 비중 확대', '인버스 ETF 헤지 고려']
            rec_class = 'bg-red'
            rec_text = '📉 매도/관망'
        else:
            action_items = ['현재 포지션 유지', '변동성 모니터링', '분할 매수 기회 포착']
            rec_class = 'bg-yellow'
            rec_text = '➡️ 중립/관망'

        actions_html = ''.join([f'<li>{a}</li>' for a in action_items])

        # 동적 리스크 경고 생성
        risk_warnings = []
        warnings = data.get('warnings', [])
        risk_score = data.get('risk_score', 0)
        bubble_risk = data.get('bubble_risk', {})
        market_quality = data.get('market_quality', {})

        # 경고 메시지에서 추출
        for w in warnings[:2]:
            if isinstance(w, str):
                risk_warnings.append(w[:50])
            elif isinstance(w, dict):
                risk_warnings.append(w.get('message', '')[:50])

        # 버블 리스크
        if isinstance(bubble_risk, dict):
            bubble_status = bubble_risk.get('overall_status', '')
            if bubble_status and bubble_status not in ['NONE', 'N/A']:
                risk_warnings.append(f"버블 리스크: {bubble_status}")

        # 리스크 점수 기반
        if risk_score > 50:
            risk_warnings.append(f"리스크 점수 상승: {risk_score:.1f}/100")

        # 유동성 리스크
        if isinstance(market_quality, dict):
            illiquid = market_quality.get('illiquid_tickers', [])
            if illiquid:
                risk_warnings.append(f"유동성 부족 자산: {len(illiquid)}개")

        # 기본 경고 (데이터 없을 경우)
        if not risk_warnings:
            risk_warnings = [
                '시장 변동성 상시 모니터링 필요',
                '포지션 크기 적정 유지 권고',
                '손절 라인 사전 설정 권장'
            ]

        warnings_html = ''.join([f'<li>{w}</li>' for w in risk_warnings[:3]])

        return f'''
<div class="card" style="margin-bottom: 24px; border: 2px solid var(--accent-blue);">
    <div class="card-header">
        <span class="card-title">✅ 최종 제안</span>
        <span class="metric-badge {rec_class}" style="font-size: 1.1rem;">{rec_text}</span>
    </div>
    <div class="grid grid-2">
        <div>
            <p style="margin-bottom: 16px;">
                <span style="font-size: 1.2rem; font-weight: 700;">{recommendation}</span>
                <span class="text-muted" style="margin-left: 8px;">(신뢰도: {confidence:.0f}%)</span>
            </p>
            <h4 style="margin-bottom: 8px;">📌 액션 아이템</h4>
            <ul style="margin-left: 20px;">
                {actions_html}
            </ul>
        </div>
        <div style="background: var(--accent-yellow-bg); padding: 16px; border-radius: 8px;">
            <h4 style="margin-bottom: 8px; color: var(--accent-yellow);">⚠️ 리스크 경고</h4>
            <ul style="margin-left: 20px; font-size: 0.9rem;">
                {warnings_html}
            </ul>
        </div>
    </div>
</div>'''

    def _generate_ai_analysis_section(self) -> str:
        """AI 종합 분석"""
        section = self.ai_report_sections.get('section_9', {})
        content = section.get('content', '')

        if not content:
            # 현재 데이터 기반 동적 분석 생성
            data = self.integrated_data
            regime = data.get('regime', {})
            regime_type = regime.get('regime', 'Unknown') if isinstance(regime, dict) else str(regime)
            risk_score = data.get('risk_score', 0)
            confidence = data.get('confidence', 0)
            if confidence <= 1:
                confidence *= 100
            recommendation = data.get('final_recommendation', 'NEUTRAL')

            # 유동성 정보
            fred = data.get('fred_summary', {})
            liquidity_regime = fred.get('liquidity_regime', 'N/A')

            # 리스크 레벨 텍스트
            if risk_score < 30:
                risk_text = "매우 낮은 위험도로 적극적 투자 가능"
            elif risk_score < 50:
                risk_text = "낮은 위험도로 균형 잡힌 투자 가능"
            elif risk_score < 70:
                risk_text = "중간 수준의 리스크, 신중한 접근 권장"
            else:
                risk_text = "높은 리스크 환경, 방어적 포지션 권고"

            # 포트폴리오 권고 생성
            if 'BULL' in recommendation.upper():
                stock_range, focus = "60-70%", "성장주/소형주 비중 증대"
            elif 'BEAR' in recommendation.upper():
                stock_range, focus = "30-40%", "방어주/배당주 중심"
            else:
                stock_range, focus = "45-55%", "균형 잡힌 섹터 배분"

            content = f"""현재 시장은 **{regime_type}** 레짐으로 분석됩니다.

### 핵심 지표 분석
- **시장 레짐**: {regime_type} - 현재 시장 상태 반영
- **유동성 환경**: {liquidity_regime}
- **리스크 점수**: {risk_score:.1f}점 - {risk_text}
- **AI 신뢰도**: {confidence:.0f}% - 분석 결과의 안정성

### 투자자 유형별 권고
- **보수적**: 주식 {int(float(stock_range.split('-')[0])*0.8)}-{int(float(stock_range.split('-')[1].replace('%',''))*0.8)}%, 대형 우량주 중심
- **적극적**: 주식 {stock_range}, {focus}
- **기관**: 전술적 자산배분 조정, 리스크 패리티 고려"""

        # Markdown to HTML
        html_content = content
        html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = re.sub(r'### (.+)', r'<h3>\1</h3>', html_content)
        html_content = re.sub(r'## (.+)', r'<h2>\1</h2>', html_content)
        html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
        html_content = html_content.replace('\n\n', '</p><p>').replace('\n', '<br>')
        html_content = f'<p>{html_content}</p>'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🤖 AI 종합 분석 (Claude)</span>
    </div>
    <div class="ai-analysis">
        {html_content}
    </div>
</div>'''

    def _generate_operational_decision(self) -> str:
        """운용 의사결정 시스템 섹션 - Operational Engine 결과 시각화 (상세 버전)"""
        op_report = self._safe_get(self.integrated_data, 'operational_report', default={})

        if not op_report:
            return '''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🎯 운용 의사결정 시스템</span>
        <span class="badge neutral">데이터 없음</span>
    </div>
    <p style="color: var(--text-secondary);">Operational Engine 결과가 없습니다. <code>python main.py</code>를 실행하세요.</p>
</div>'''

        # =============================================
        # 1. HOLD POLICY (홀드 판단 과정)
        # =============================================
        hold_policy = op_report.get('hold_policy', {})
        is_hold = hold_policy.get('is_hold', False)
        hold_conditions = hold_policy.get('hold_conditions', [])

        hold_conditions_html = ''
        triggered_hold_html = ''
        for cond in hold_conditions:
            triggered = cond.get('is_triggered', False)
            if isinstance(triggered, str):
                triggered = triggered.lower() == 'true'
            status_icon = '🔴' if triggered else '🟢'
            status_class = 'accent-red' if triggered else 'accent-green'
            hold_conditions_html += f'''
            <tr>
                <td>{cond.get('priority', '-')}</td>
                <td>{cond.get('condition_name', 'N/A')}</td>
                <td style="color: var(--{status_class}); font-weight: 600;">{status_icon} {'TRIGGERED' if triggered else 'PASS'}</td>
                <td><code>{cond.get('current_value', 'N/A')}</code></td>
                <td>{cond.get('threshold', 'N/A')}</td>
            </tr>'''
            if triggered:
                triggered_hold_html += f'''
                <div style="background: var(--accent-red-bg); padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid var(--accent-red);">
                    <strong>{cond.get('condition_name')}</strong>: {cond.get('description', '')}
                    <div style="font-size: 0.85rem; margin-top: 4px;">
                        현재: <code>{cond.get('current_value')}</code> → 필요: <code>{cond.get('threshold')}</code>
                    </div>
                </div>'''

        # =============================================
        # 2. DECISION POLICY (의사결정 규칙)
        # =============================================
        policy = op_report.get('decision_policy', {})
        final_stance = policy.get('final_stance', 'N/A')
        stance_class = 'bullish' if final_stance == 'BULLISH' else ('bearish' if final_stance == 'BEARISH' else 'neutral')
        constraint_status = policy.get('constraint_status_input', policy.get('constraints_ok', 'OK'))
        constraints_ok = constraint_status in ('OK', 'REPAIRED', True)
        client_profile = policy.get('client_profile_status_input', policy.get('client_profile', 'N/A'))
        applied_rules = policy.get('applied_rules', [])
        reason_codes = policy.get('reason_codes', [])
        rule_evaluation_log = policy.get('rule_evaluation_log', [])

        # Rule Evaluation Log HTML
        rule_eval_html = ''
        for rule in rule_evaluation_log:
            result = rule.get('result', '')
            is_passed = 'PASSED' in result or 'NOT_TRIGGERED' in result
            result_class = 'accent-green' if is_passed else 'accent-red'
            result_icon = '✅' if is_passed else '⛔'
            rule_eval_html += f'''
            <tr>
                <td><code>{rule.get('rule', 'N/A')}</code></td>
                <td style="font-size: 0.8rem;">{rule.get('condition', 'N/A')}</td>
                <td><code>{rule.get('input', 'N/A')}</code></td>
                <td style="color: var(--{result_class}); font-weight: 600;">{result_icon} {result}</td>
            </tr>'''

        # Applied Rules HTML
        rules_html = ''
        for rule in applied_rules[:5]:
            if isinstance(rule, dict):
                rule_name = rule.get('rule', '')
            else:
                rule_name = str(rule)
            result_class = 'accent-green' if 'BULLISH' in rule_name else ('accent-red' if 'HOLD' in rule_name or 'BEARISH' in rule_name else 'accent-blue')
            rules_html += f'<div style="padding: 6px 0; border-bottom: 1px solid var(--border);"><span style="color: var(--{result_class}); font-weight: 600;">{rule_name}</span></div>'

        reason_html = ', '.join([f'<code>{c}</code>' for c in reason_codes]) if reason_codes else '<span style="color: var(--text-muted);">없음</span>'

        # =============================================
        # 3. SCORE DEFINITIONS (단일 Canonical 점수)
        # =============================================
        scores = op_report.get('score_definitions', {})
        canonical_risk = scores.get('canonical_risk_score', 0)
        risk_level = scores.get('risk_level', 'MEDIUM')
        risk_level_class = 'accent-green' if risk_level == 'LOW' else ('accent-red' if risk_level == 'HIGH' else 'accent-yellow')

        aux_sub = scores.get('auxiliary_sub_scores', {})
        aux_scores = {}
        aux_sources = {}
        for key, val in aux_sub.items():
            if isinstance(val, dict):
                aux_scores[key] = val.get('value', 0)
                aux_sources[key] = val.get('source', 'N/A')
            else:
                aux_scores[key] = val
                aux_sources[key] = 'N/A'

        aux_html = ''
        for key in ['base_risk_score', 'microstructure_adjustment', 'bubble_risk_adjustment', 'extended_data_adjustment']:
            val = aux_scores.get(key, 0)
            source = aux_sources.get(key, 'N/A')
            val_str = f"{val:+.1f}" if 'adjustment' in key else f"{val:.1f}"
            aux_html += f'''
            <tr>
                <td>{key}</td>
                <td style="font-weight: 600;">{val_str}</td>
                <td style="font-size: 0.8rem; color: var(--text-secondary);">{source}</td>
            </tr>'''

        # Calculate formula
        base = aux_scores.get('base_risk_score', 0)
        micro = aux_scores.get('microstructure_adjustment', 0)
        bubble = aux_scores.get('bubble_risk_adjustment', 0)
        extended = aux_scores.get('extended_data_adjustment', 0)

        # =============================================
        # 4. CONSTRAINT REPAIR (제약조건 수리)
        # =============================================
        repair = op_report.get('constraint_repair', {})
        repair_ok = repair.get('constraints_satisfied', repair.get('constraints_ok', True))
        force_hold = repair.get('force_hold', False)
        force_hold_reason = repair.get('force_hold_reason', '')
        violations = repair.get('violations_found', repair.get('violations', []))
        repair_actions = repair.get('repair_actions', [])
        asset_class_comparison = repair.get('asset_class_comparison', [])

        violations_html = ''
        for v in violations:
            current = v.get('current_value', v.get('current_weight', 0))
            limit_val = v.get('limit_value', v.get('limit', 0))
            violations_html += f'''
            <tr>
                <td>{v.get('asset_class', 'N/A')}</td>
                <td style="color: var(--accent-red);">{v.get('violation_type', 'N/A')}</td>
                <td>{current:.1%}</td>
                <td>{limit_val:.1%}</td>
            </tr>'''

        comparison_html = ''
        for c in asset_class_comparison:
            status = c.get('status', 'OK')
            status_icon = '✅' if status == 'OK' else '⚠️'
            comparison_html += f'''
            <tr>
                <td>{c.get('asset_class', 'N/A')}</td>
                <td>{c.get('original_weight', 0):.1%}</td>
                <td>{c.get('repaired_weight', 0):.1%}</td>
                <td style="color: var(--{'accent-green' if c.get('delta', 0) >= 0 else 'accent-red'});">{c.get('delta', 0):+.1%}</td>
                <td>{status_icon} {status}</td>
            </tr>'''

        # =============================================
        # 5. REBALANCE PLAN (리밸런싱 계획)
        # =============================================
        rebalance = op_report.get('rebalance_plan', {})
        execution = rebalance.get('execution', {})
        should_execute = execution.get('should_execute', rebalance.get('should_execute', False))
        not_executed_reason = execution.get('not_executed_reason', rebalance.get('not_executed_reason', ''))

        trigger = rebalance.get('trigger', {})
        trigger_type = trigger.get('type', rebalance.get('trigger_type', 'MANUAL'))

        summary = rebalance.get('summary', {})
        turnover = summary.get('total_turnover', rebalance.get('turnover', 0))
        buy_count = summary.get('buy_count', 0)
        sell_count = summary.get('sell_count', 0)

        cost_breakdown = rebalance.get('cost_breakdown', {})
        commission = cost_breakdown.get('commission', 0)
        spread = cost_breakdown.get('spread', 0)
        market_impact = cost_breakdown.get('market_impact', 0)
        total_cost = cost_breakdown.get('total', rebalance.get('est_total_cost', 0))

        approval = rebalance.get('approval', {})
        requires_approval = approval.get('requires_human_approval', rebalance.get('requires_approval', False))
        approval_reason = approval.get('approval_reason', '')

        trades = rebalance.get('trades', [])
        trades_html = ''
        for i, t in enumerate(trades[:10], 1):
            action = t.get('action', 'HOLD')
            action_class = 'accent-green' if action == 'BUY' else ('accent-red' if action == 'SELL' else 'text-secondary')
            trades_html += f'''
            <tr>
                <td>{i}</td>
                <td>{t.get('ticker', 'N/A')}</td>
                <td style="color: var(--{action_class}); font-weight: 600;">{action}</td>
                <td>{t.get('current_weight', 0):.2%}</td>
                <td>{t.get('target_weight', 0):.2%}</td>
                <td style="color: var(--{'accent-green' if t.get('delta_weight', t.get('delta', 0)) >= 0 else 'accent-red'});">{t.get('delta_weight', t.get('delta', 0)):+.2%}</td>
                <td>{t.get('estimated_cost', t.get('est_cost', 0)):.4f}</td>
            </tr>'''

        # =============================================
        # BUILD FINAL HTML
        # =============================================
        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🎯 운용 의사결정 시스템 (Operational Engine)</span>
        <span class="status-badge {stance_class}">{final_stance}</span>
    </div>

    <!-- SECTION 1: HOLD POLICY -->
    <div style="background: var(--{'accent-red-bg' if is_hold else 'accent-green-bg'}); padding: 16px; border-radius: 8px; margin-bottom: 20px; border: 2px solid var(--{'accent-red' if is_hold else 'accent-green'});">
        <h4 style="margin-bottom: 12px; color: var(--{'accent-red' if is_hold else 'accent-green'});">
            {'⛔ HOLD TRIGGERED - 거래 중단' if is_hold else '✅ HOLD 조건 통과 - 거래 진행 가능'}
        </h4>

        <table class="table" style="width: 100%; margin-bottom: 12px;">
            <thead>
                <tr><th>Priority</th><th>Condition</th><th>Status</th><th>Current</th><th>Required</th></tr>
            </thead>
            <tbody>{hold_conditions_html}</tbody>
        </table>

        {f'<div style="margin-top: 12px;"><strong>🚨 Triggered Conditions (Conflict Resolution):</strong>{triggered_hold_html}</div>' if triggered_hold_html else ''}
    </div>

    <!-- SECTION 2: DECISION POLICY -->
    <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin-bottom: 12px; color: var(--accent-blue);">📋 Decision Policy (규칙 기반 의사결정)</h4>

        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px;">
            <div style="flex: 1; min-width: 200px;">
                <div style="padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span>final_stance:</span>
                    <span class="status-badge {stance_class}" style="padding: 4px 12px; font-size: 0.85rem; margin-left: 8px;">{final_stance}</span>
                </div>
                <div style="padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span>constraints_ok:</span>
                    <span style="color: var(--{'accent-green' if constraints_ok else 'accent-red'}); font-weight: 600; margin-left: 8px;">{'✓ OK' if constraints_ok else '✗ VIOLATED'}</span>
                </div>
                <div style="padding: 8px 0;">
                    <span>client_profile:</span>
                    <span style="font-weight: 600; margin-left: 8px;">{client_profile}</span>
                </div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="padding: 8px 0;"><strong>reason_codes:</strong> {reason_html}</div>
            </div>
        </div>

        <details style="margin-top: 12px;">
            <summary style="cursor: pointer; font-weight: 600; color: var(--accent-blue);">📜 Rule Evaluation Log (클릭하여 펼치기)</summary>
            <div style="margin-top: 12px; overflow-x: auto;">
                <table class="table" style="width: 100%;">
                    <thead><tr><th>Rule</th><th>Condition</th><th>Input</th><th>Result</th></tr></thead>
                    <tbody>{rule_eval_html}</tbody>
                </table>
            </div>
        </details>

        <div style="margin-top: 12px;">
            <strong>Applied Rules:</strong>
            {rules_html if rules_html else '<span style="color: var(--text-muted);">없음</span>'}
        </div>
    </div>

    <!-- SECTION 3: SCORE DEFINITIONS (Canonical Only) -->
    <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin-bottom: 12px; color: var(--accent-purple);">📊 Score Definitions (단일 Canonical 점수)</h4>

        <div style="background: var(--accent-purple-bg); padding: 16px; border-radius: 8px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 0.9rem; color: var(--text-secondary);">의사결정에 사용되는 유일한 점수</div>
            <div style="font-size: 2rem; font-weight: 700; color: var(--accent-purple);">{canonical_risk:.1f} / 100</div>
            <div style="font-size: 1rem; color: var(--{risk_level_class}); font-weight: 600;">{risk_level}</div>
        </div>

        <div style="background: var(--bg-secondary); padding: 12px; border-radius: 6px; margin-bottom: 12px;">
            <strong>⚠️ Important:</strong> 다른 점수들은 <strong>참고용</strong>입니다. 모든 규칙은 canonical_risk_score만 사용합니다.
        </div>

        <details>
            <summary style="cursor: pointer; font-weight: 600;">🔍 Auxiliary Sub-Scores (참고용)</summary>
            <table class="table" style="width: 100%; margin-top: 12px;">
                <thead><tr><th>Component</th><th>Value</th><th>Source</th></tr></thead>
                <tbody>{aux_html}</tbody>
            </table>
            <div style="margin-top: 12px; padding: 12px; background: var(--bg-secondary); border-radius: 6px; font-family: monospace; font-size: 0.85rem;">
                <strong>Formula:</strong><br>
                canonical = {base:.1f} + ({micro:+.1f}) + ({bubble:+.1f}) + ({extended:+.1f}) = <strong>{canonical_risk:.1f}</strong>
            </div>
        </details>
    </div>

    <!-- SECTION 4: CONSTRAINT REPAIR -->
    <div style="background: var(--{'accent-red-bg' if force_hold else ('accent-green-bg' if repair_ok else 'accent-yellow-bg')}); padding: 16px; border-radius: 8px; margin-bottom: 20px; border: 1px solid var(--{'accent-red' if force_hold else ('accent-green' if repair_ok else 'accent-yellow')});">
        <h4 style="margin-bottom: 12px; color: var(--{'accent-red' if force_hold else ('accent-green' if repair_ok else 'accent-yellow')});">
            🔧 Constraint Repair {'⛔ FORCE HOLD' if force_hold else ('✅ SATISFIED' if repair_ok else '🔄 REPAIRED')}
        </h4>

        {f'<div style="background: var(--accent-red-bg); padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid var(--accent-red);"><strong>Force HOLD Reason:</strong> {force_hold_reason}</div>' if force_hold else ''}

        {f'''
        <details open>
            <summary style="cursor: pointer; font-weight: 600;">Violations Detected</summary>
            <table class="table" style="width: 100%; margin-top: 12px;">
                <thead><tr><th>Asset Class</th><th>Type</th><th>Current</th><th>Limit</th></tr></thead>
                <tbody>{violations_html}</tbody>
            </table>
        </details>
        ''' if violations_html else ''}

        {f'''
        <details style="margin-top: 12px;">
            <summary style="cursor: pointer; font-weight: 600;">before_weights vs after_weights</summary>
            <table class="table" style="width: 100%; margin-top: 12px;">
                <thead><tr><th>Asset Class</th><th>Before</th><th>After</th><th>Delta</th><th>Status</th></tr></thead>
                <tbody>{comparison_html}</tbody>
            </table>
        </details>
        ''' if comparison_html else ''}
    </div>

    <!-- SECTION 5: REBALANCE PLAN -->
    <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin-bottom: 12px; color: var(--accent-cyan);">
            💰 Rebalance Plan {'✅ EXECUTE' if should_execute else '⏸️ NOT EXECUTED'}
        </h4>

        {f'<div style="background: var(--accent-yellow-bg); padding: 12px; border-radius: 6px; margin-bottom: 12px;">ℹ️ {not_executed_reason}</div>' if not should_execute else ''}

        <div class="grid grid-2" style="margin-bottom: 16px;">
            <div>
                <table class="table" style="width: 100%;">
                    <tr><td><strong>turnover</strong></td><td>{turnover:.2%}</td></tr>
                    <tr><td><strong>trigger_type</strong></td><td>{trigger_type}</td></tr>
                    <tr><td><strong>requires_approval</strong></td><td style="color: var(--{'accent-red' if requires_approval else 'accent-green'});">{'⚠️ YES' if requires_approval else '✅ NO'}</td></tr>
                    <tr><td>Buy Orders</td><td>{buy_count}</td></tr>
                    <tr><td>Sell Orders</td><td>{sell_count}</td></tr>
                </table>
            </div>
            <div>
                <table class="table" style="width: 100%;">
                    <tr><td>Commission</td><td>{commission:.4f}</td></tr>
                    <tr><td>Spread</td><td>{spread:.4f}</td></tr>
                    <tr><td>Market Impact</td><td>{market_impact:.4f}</td></tr>
                    <tr><td><strong>est_total_cost</strong></td><td><strong>{total_cost:.4f}</strong></td></tr>
                </table>
            </div>
        </div>

        {f'<div style="background: var(--accent-red-bg); padding: 12px; border-radius: 6px; margin-bottom: 12px;">⚠️ <strong>Human Approval Required:</strong> {approval_reason}</div>' if requires_approval else ''}

        {f'''
        <details>
            <summary style="cursor: pointer; font-weight: 600;">📝 Trade List ({len(trades)} trades)</summary>
            <table class="table" style="width: 100%; margin-top: 12px;">
                <thead><tr><th>#</th><th>Ticker</th><th>Action</th><th>Current</th><th>Target</th><th>Delta</th><th>Cost</th></tr></thead>
                <tbody>{trades_html}</tbody>
            </table>
        </details>
        ''' if trades_html else ''}
    </div>
</div>'''

    def _generate_footer(self) -> str:
        """푸터"""
        return f'''
<div class="footer">
    <p class="footer-brand">EIMAS v2.2.0 (Economic Intelligence Multi-Agent System)</p>
    <p>본 보고서는 AI 알고리즘에 의해 자동 생성되었으며, 투자 판단의 참고 자료로만 활용하시기 바랍니다.</p>

    <div class="disclaimer">
        <strong>⚠️ 면책조항:</strong> 본 리포트는 정보 제공 목적으로만 작성되었으며, 투자 권유나 매매 추천을 구성하지 않습니다.
        AI 모델의 분석은 과거 데이터에 기반하며, 미래 수익을 보장하지 않습니다.
        모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
    </div>

    <p style="margin-top: 16px; color: var(--text-muted);">
        Data Sources: FRED, Yahoo Finance, Perplexity AI, OpenAI GPT-4o, Anthropic Claude
    </p>
</div>'''


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EIMAS Final Report Agent v2.0 - 최종 HTML/PDF 리포트 생성'
    )
    parser.add_argument('--user', '-u', type=str, default='EIMAS', help='보고서 작성자/수신자 이름')
    parser.add_argument('--output', '-o', type=str, default='outputs', help='출력 디렉토리')
    parser.add_argument('--pdf', '-p', action='store_true', help='PDF도 함께 생성')

    args = parser.parse_args()

    print("=" * 60)
    print("EIMAS Final Report Agent v2.0")
    print("=" * 60)

    agent = FinalReportAgent(output_dir=args.output, user_name=args.user)

    print("\n[1/3] Loading latest data...")
    agent.load_latest_data()

    print("\n[2/3] Generating HTML report...")
    html = agent.generate_report()
    print(f"  Generated {len(html):,} characters")

    print("\n[3/3] Saving report...")
    output_path = agent.save_report()

    # PDF 변환 (옵션)
    pdf_path = None
    if args.pdf:
        print("\n[4/4] Converting to PDF...")
        pdf_path = agent.save_pdf(output_path)

    print("\n" + "=" * 60)
    print(f"HTML: {output_path}")
    if pdf_path:
        print(f"PDF:  {pdf_path}")
    print("=" * 60)

    return output_path


if __name__ == '__main__':
    main()
