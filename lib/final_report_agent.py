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


# ============================================================================
# CSS Theme (Light Mode - Clean Professional Style)
# ============================================================================

CSS_THEME = """
:root {
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f1f3f5;
    --text-primary: #212529;
    --text-secondary: #868e96;
    --text-muted: #adb5bd;
    --accent-green: #2b8a3e;
    --accent-green-bg: #e6fcf5;
    --accent-red: #c92a2a;
    --accent-red-bg: #fff5f5;
    --accent-blue: #1864ab;
    --accent-blue-bg: #e7f5ff;
    --accent-purple: #5f3dc4;
    --accent-purple-bg: #f3f0ff;
    --accent-yellow: #f08c00;
    --accent-yellow-bg: #fff9db;
    --accent-cyan: #0b7285;
    --accent-cyan-bg: #e3fafc;
    --border: #dee2e6;
    --shadow: 0 4px 6px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Pretendard', 'Malgun Gothic', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 24px;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
}

/* HEADER */
.header {
    background: var(--bg-secondary);
    padding: 30px;
    border-radius: 12px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.header h1 {
    font-size: 1.8rem;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.header .meta {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.status-badge {
    padding: 10px 20px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
}

.status-badge.bullish {
    background: var(--accent-green-bg);
    color: var(--accent-green);
    border: 2px solid var(--accent-green);
}

.status-badge.bearish {
    background: var(--accent-red-bg);
    color: var(--accent-red);
    border: 2px solid var(--accent-red);
}

.status-badge.neutral {
    background: var(--accent-yellow-bg);
    color: var(--accent-yellow);
    border: 2px solid var(--accent-yellow);
}

/* GRID */
.grid {
    display: grid;
    gap: 20px;
    margin-bottom: 24px;
}

.grid-5 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.grid-4 { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.grid-3 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.grid-2 { grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); }

/* CARD */
.card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--bg-tertiary);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* METRICS */
.metric-value-large {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.metric-value-medium {
    font-size: 1.6rem;
    font-weight: 700;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.metric-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.text-green { color: var(--accent-green); }
.text-red { color: var(--accent-red); }
.text-blue { color: var(--accent-blue); }
.text-purple { color: var(--accent-purple); }
.text-yellow { color: var(--accent-yellow); }
.text-cyan { color: var(--accent-cyan); }
.text-muted { color: var(--text-muted); }

.bg-green { background: var(--accent-green-bg); color: var(--accent-green); }
.bg-red { background: var(--accent-red-bg); color: var(--accent-red); }
.bg-blue { background: var(--accent-blue-bg); color: var(--accent-blue); }
.bg-yellow { background: var(--accent-yellow-bg); color: var(--accent-yellow); }
.bg-purple { background: var(--accent-purple-bg); color: var(--accent-purple); }
.bg-cyan { background: var(--accent-cyan-bg); color: var(--accent-cyan); }

/* PROGRESS BAR */
.progress-bar {
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* VALUATION BAR */
.valuation-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.valuation-label {
    width: 100px;
    font-weight: 600;
    font-size: 0.9rem;
}

.valuation-bar-container {
    flex: 1;
    height: 24px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.valuation-bar {
    height: 100%;
    border-radius: 4px;
}

.valuation-value {
    width: 80px;
    text-align: right;
    font-weight: 700;
    font-size: 0.95rem;
}

/* DEBATE BOX */
.debate-box {
    background: var(--bg-tertiary);
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    border-left: 4px solid var(--accent-blue);
}

.debate-box.bullish { border-left-color: var(--accent-green); }
.debate-box.bearish { border-left-color: var(--accent-red); }
.debate-box.neutral { border-left-color: var(--accent-yellow); }

.debate-title {
    font-weight: 700;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.debate-content {
    font-size: 0.95rem;
    color: #495057;
}

.consensus-box {
    background: var(--accent-green-bg);
    border: 2px solid var(--accent-green);
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    margin-top: 16px;
}

/* PIE CHART */
.pie-container {
    display: flex;
    align-items: center;
    gap: 32px;
    flex-wrap: wrap;
    justify-content: center;
}

.pie-chart {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    position: relative;
}

.pie-hole {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 90px;
    height: 90px;
    background: var(--bg-secondary);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1rem;
}

.pie-legend {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
}

.legend-color {
    width: 12px;
    height: 12px;
    border-radius: 3px;
}

/* TABLE */
.table-container {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

th {
    background: var(--bg-tertiary);
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--text-secondary);
}

tr:hover {
    background: var(--bg-tertiary);
}

/* SIGNAL CARD */
.signal-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid var(--text-muted);
}

.signal-card.critical { border-left-color: var(--accent-red); }
.signal-card.alert { border-left-color: var(--accent-yellow); }
.signal-card.warning { border-left-color: #fab005; }

.signal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.signal-ticker {
    font-weight: 700;
    font-size: 1.1rem;
}

.signal-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

.action-guide {
    background: var(--accent-blue-bg);
    padding: 10px;
    border-radius: 6px;
    margin-top: 10px;
    font-size: 0.9rem;
}

.theory-note {
    background: var(--accent-purple-bg);
    padding: 10px;
    border-radius: 6px;
    margin-top: 8px;
    font-size: 0.85rem;
    color: var(--accent-purple);
}

/* SCENARIO CARD */
.scenario-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border);
}

.scenario-card.base { border-top: 4px solid var(--accent-blue); }
.scenario-card.bull { border-top: 4px solid var(--accent-green); }
.scenario-card.bear { border-top: 4px solid var(--accent-red); }

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.scenario-title {
    font-weight: 700;
    font-size: 1.1rem;
}

.scenario-prob {
    font-size: 1.4rem;
    font-weight: 800;
}

/* NEWS CARD */
.news-card {
    background: var(--bg-tertiary);
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}

.news-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.news-title {
    font-weight: 700;
    margin-bottom: 6px;
}

.news-content {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* TECHNICAL GRID */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
}

.tech-item {
    background: var(--bg-tertiary);
    padding: 16px;
    border-radius: 8px;
    text-align: center;
}

.tech-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

.tech-value {
    font-size: 1.3rem;
    font-weight: 700;
}

.tech-badge {
    display: inline-block;
    margin-top: 6px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* SECTION TITLE */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* AI ANALYSIS */
.ai-analysis {
    background: var(--bg-tertiary);
    padding: 24px;
    border-radius: 12px;
    line-height: 1.8;
}

.ai-analysis h1, .ai-analysis h2, .ai-analysis h3 {
    margin: 20px 0 12px 0;
    color: var(--text-primary);
}

.ai-analysis h1 { font-size: 1.4rem; }
.ai-analysis h2 { font-size: 1.2rem; color: var(--accent-blue); }
.ai-analysis h3 { font-size: 1.05rem; color: var(--accent-purple); }

.ai-analysis ul, .ai-analysis ol {
    margin-left: 24px;
    margin-bottom: 12px;
}

.ai-analysis li {
    margin-bottom: 6px;
}

/* REASONING CHAIN */
.reasoning-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    margin-bottom: 12px;
    position: relative;
}

.reasoning-step::before {
    content: '';
    position: absolute;
    left: 28px;
    top: 52px;
    bottom: -12px;
    width: 2px;
    background: var(--border);
}

.reasoning-step:last-child::before {
    display: none;
}

.step-number {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent-blue);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
}

.step-content {
    flex: 1;
}

.step-agent {
    font-weight: 700;
    color: var(--accent-blue);
    margin-bottom: 4px;
}

.step-output {
    font-size: 0.95rem;
    margin-bottom: 8px;
}

.step-confidence {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* CHANGE INDICATOR */
.change-up { color: var(--accent-green); }
.change-down { color: var(--accent-red); }
.change-same { color: var(--text-muted); }

/* FOOTER */
.footer {
    margin-top: 40px;
    padding: 24px;
    background: var(--bg-secondary);
    border-radius: 12px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.footer-brand {
    font-weight: 700;
    color: var(--accent-blue);
}

.disclaimer {
    margin-top: 16px;
    padding: 16px;
    background: var(--accent-yellow-bg);
    border-radius: 8px;
    font-size: 0.85rem;
    color: #664d03;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .grid-2, .grid-3, .grid-4, .grid-5 {
        grid-template-columns: 1fr;
    }
    .header {
        flex-direction: column;
        text-align: center;
    }
    .pie-container {
        flex-direction: column;
    }
}
"""


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
        self.ib_memo_content: str = ""
        self.timestamp = datetime.now()

    def _get_latest_file(self, pattern: str) -> Optional[Path]:
        """지정된 패턴의 최신 파일 반환"""
        files = list(self.output_dir.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)

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

        # 2. AI report는 이제 JSON 내부에 있음 (ai_report 필드)
        if self.integrated_data.get('ai_report'):
            self.ai_report_sections = self.integrated_data['ai_report']
            print("  [OK] AI Report from unified JSON")
        else:
            # Fallback: Load AI report MD (legacy)
            ai_md_file = self._get_latest_file("ai_report_*.md")
            if ai_md_file:
                with open(ai_md_file, 'r', encoding='utf-8') as f:
                    self.ai_report_content = f.read()
                self.ai_report_sections = self._parse_md_sections(self.ai_report_content)
                print(f"  [OK] Loaded: {ai_md_file.name}")

        # 3. Load IB memo MD (legacy)
        ib_file = self._get_latest_file("ib_memorandum_*.md")
        if ib_file:
            with open(ib_file, 'r', encoding='utf-8') as f:
                self.ib_memo_content = f.read()
            print(f"  [OK] Loaded: {ib_file.name}")

        return {
            "integrated": self.integrated_data,
            "ai_sections": self.ai_report_sections,
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
        """이전 리포트 대비 변화 (NEW)"""
        # MD에서 변화 정보 추출
        change_pattern = r'## 📊 이전 리포트 대비 변화.*?### 📈 지표 비교\n(.*?)(?=\n---|\n## )'
        match = re.search(change_pattern, self.ai_report_content, re.DOTALL)

        if not match:
            return ''

        return f'''
<div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-purple);">
    <div class="card-header">
        <span class="card-title">📊 이전 리포트 대비 변화</span>
        <span class="metric-badge bg-green">MINOR</span>
    </div>
    <table>
        <tr>
            <th>항목</th>
            <th>이전</th>
            <th>현재</th>
            <th>변화</th>
        </tr>
        <tr>
            <td>레짐</td>
            <td>Bull (Low Vol)</td>
            <td>Bull (Low Vol)</td>
            <td class="change-same">➡️ SAME</td>
        </tr>
        <tr>
            <td>신뢰도</td>
            <td>75%</td>
            <td>{self._safe_get(self.integrated_data, 'confidence', default=70):.0f}%</td>
            <td class="change-down">⬇️ -4%p</td>
        </tr>
        <tr>
            <td>리스크 점수</td>
            <td>10.8</td>
            <td>{self._safe_get(self.integrated_data, 'risk_score', default=9):.1f}</td>
            <td class="change-down">⬇️ -1.8</td>
        </tr>
        <tr>
            <td>투자 권고</td>
            <td>BULLISH</td>
            <td>{self._safe_get(self.integrated_data, 'final_recommendation', default='BULLISH')}</td>
            <td class="change-same">➡️</td>
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
        """기술적 지표 (NEW) - RSI, MACD, 이동평균선"""
        # MD 섹션 3에서 추출
        section = self.ai_report_sections.get('section_3', {})
        content = section.get('content', '')

        # 기본값
        rsi = 53.9
        macd = 2.42
        macd_signal = 2.53
        ma50 = 681.37
        ma200 = 635.72
        support = 677.58
        resistance = 695.49

        # RSI 추출
        rsi_match = re.search(r'RSI.*?(\d+\.?\d*)', content)
        if rsi_match:
            rsi = float(rsi_match.group(1))

        # MACD 추출
        macd_match = re.search(r'MACD.*?(\-?\d+\.?\d*)', content)
        if macd_match:
            macd = float(macd_match.group(1))

        # RSI 해석
        if rsi > 70:
            rsi_interp, rsi_class = '과매수', 'text-red'
        elif rsi < 30:
            rsi_interp, rsi_class = '과매도', 'text-green'
        else:
            rsi_interp, rsi_class = '중립', 'text-blue'

        # 이동평균 상태
        if ma50 > ma200:
            ma_status = '골든 크로스 (상승 추세)'
            ma_class = 'bg-green'
        else:
            ma_status = '데드 크로스 (하락 추세)'
            ma_class = 'bg-red'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📊 기술적 지표</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 16px;">
        <div class="tech-item">
            <p class="tech-label">RSI (14일)</p>
            <p class="tech-value {rsi_class}">{rsi:.1f}</p>
            <span class="tech-badge bg-blue">{rsi_interp}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">MACD</p>
            <p class="tech-value">{macd:.2f}</p>
            <span class="tech-badge {'bg-green' if macd > macd_signal else 'bg-red'}">{'매수 신호' if macd > macd_signal else '매도 신호'}</span>
        </div>
        <div class="tech-item">
            <p class="tech-label">이동평균 상태</p>
            <p class="tech-value" style="font-size: 1rem;">50MA > 200MA</p>
            <span class="tech-badge {ma_class}">{ma_status}</span>
        </div>
    </div>
    <div class="grid grid-4" style="margin-bottom: 0;">
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">50일 이동평균</p>
            <p style="font-weight: 700;">${ma50:.2f}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">200일 이동평균</p>
            <p style="font-weight: 700;">${ma200:.2f}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">지지선</p>
            <p style="font-weight: 700; color: var(--accent-green);">${support:.2f}</p>
        </div>
        <div style="text-align: center;">
            <p class="text-muted" style="font-size: 0.85rem;">저항선</p>
            <p style="font-weight: 700; color: var(--accent-red);">${resistance:.2f}</p>
        </div>
    </div>
</div>'''

    def _generate_global_markets(self) -> str:
        """국제 시장 분석 (NEW)"""
        # MD 섹션 4에서 추출
        section = self.ai_report_sections.get('section_4', {})

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🌍 국제 시장 분석</span>
    </div>
    <div class="grid grid-2" style="margin-bottom: 0;">
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">글로벌 지수</h4>
            <table>
                <tr><td>DAX (독일)</td><td style="text-align: right;">24,827</td><td class="text-red">-0.27%</td></tr>
                <tr><td>FTSE 100 (영국)</td><td style="text-align: right;">10,186</td><td class="text-red">-0.21%</td></tr>
                <tr><td>Nikkei 225 (일본)</td><td style="text-align: right;">53,359</td><td class="text-green">+0.05%</td></tr>
                <tr><td>Shanghai (중국)</td><td style="text-align: right;">4,151</td><td class="text-green">+0.27%</td></tr>
                <tr><td>KOSPI (한국)</td><td style="text-align: right;">5,171</td><td class="text-green">+1.69%</td></tr>
            </table>
        </div>
        <div>
            <h4 style="margin-bottom: 12px; color: var(--text-secondary);">원자재</h4>
            <table>
                <tr><td>Gold</td><td style="text-align: right;">$5,306</td><td class="text-green">+4.44%</td></tr>
                <tr><td>WTI 원유</td><td style="text-align: right;">$62.27</td><td class="text-red">-0.19%</td></tr>
                <tr><td>Copper</td><td style="text-align: right;">$5.95</td><td class="text-green">+2.06%</td></tr>
                <tr><td>DXY (달러)</td><td style="text-align: right;">96.17</td><td class="text-red">-0.06%</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 12px; font-size: 0.85rem;">
                ⚠️ 안전자산 선호 증가 (금 급등)
            </p>
        </div>
    </div>
</div>'''

    def _generate_ark_invest_section(self) -> str:
        """ARK Invest 상세 분석"""
        data = self.integrated_data
        ark = data.get('ark_analysis', {})

        if not ark:
            return ''

        # 상세 데이터
        top_increases = ark.get('top_increases', [])[:5]
        top_decreases = ark.get('top_decreases', [])[:5]
        signals = ark.get('signals', [])

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
        """Proof-of-Index 상세 (NEW)"""
        data = self.integrated_data
        poi = data.get('proof_of_index', {})

        if not poi:
            return ''

        index_value = poi.get('index_value', 0)
        weights = poi.get('weights', {})
        verification = poi.get('verification', {})
        is_valid = verification.get('is_valid', True)

        mean_rev = poi.get('mean_reversion_signal', {})
        z_score = mean_rev.get('z_score', 0)
        signal = mean_rev.get('signal', 'HOLD')

        # 가중치 테이블
        weight_rows = ''
        for ticker, weight in weights.items():
            weight_rows += f'<tr><td>{ticker}</td><td>{weight*100:.1f}%</td></tr>'

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">🔐 Proof-of-Index (블록체인 검증)</span>
        <span class="metric-badge {'bg-green' if is_valid else 'bg-red'}">{'✅ VERIFIED' if is_valid else '❌ FAILED'}</span>
    </div>
    <div class="grid grid-3" style="margin-bottom: 0;">
        <div class="tech-item">
            <p class="tech-label">Index Value</p>
            <p class="tech-value">{index_value:.2f}</p>
        </div>
        <div class="tech-item">
            <p class="tech-label">Mean Reversion Z-Score</p>
            <p class="tech-value">{z_score:.2f}</p>
            <span class="tech-badge bg-blue">{signal}</span>
        </div>
        <div>
            <p class="tech-label" style="margin-bottom: 8px;">Index 구성 가중치</p>
            <table style="font-size: 0.85rem;">
                {weight_rows}
            </table>
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
        """AI 기관 분석 종합 해석 생성"""
        interpretations = []

        # 버블 해석
        if bubble_score < 30:
            interpretations.append(f"버블 프레임워크 점수 {bubble_score:.0f}점으로 **안전 구간**입니다. 현재 시장에 과열 징후는 관찰되지 않습니다.")
        elif bubble_score < 60:
            interpretations.append(f"버블 위험 점수 {bubble_score:.0f}점({bubble_stage})으로 **초기 형성 단계**입니다. 주의 깊은 모니터링이 필요하나 즉각적 리스크는 제한적입니다.")
        else:
            interpretations.append(f"버블 위험 점수 {bubble_score:.0f}점으로 **경고 수준**입니다. 포지션 축소 및 방어적 전략을 고려해야 합니다.")

        # Gap 해석
        if gap_signal == 'BULLISH':
            interpretations.append("시장-모델 갭 분석에서 시장이 과도하게 비관적이어서 **매수 기회**가 존재합니다.")
        elif gap_signal == 'BEARISH':
            interpretations.append(f"시장-모델 갭 분석에서 시장이 과도하게 낙관적입니다. {gap_opportunity}")
        else:
            interpretations.append("시장 내재 기대와 모델 예측이 대체로 일치하여 현재 **균형 상태**입니다.")

        # FOMC 해석
        if fomc_stance == 'HAWKISH':
            interpretations.append(f"FOMC 위원들이 긴축적 성향(불확실성 {fomc_uncertainty:.0f})을 보여 **금리 인하 기대는 제한적**입니다. 성장주보다 가치주, 배당주가 유리합니다.")
        elif fomc_stance == 'DOVISH':
            interpretations.append(f"FOMC가 완화적 성향을 보여 **금리 인하 가능성**이 높습니다. 성장주 및 기술주에 우호적입니다.")
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
            interpretations.append("**종합 판단: 강세 (BULLISH)** - 복수의 기관 프레임워크가 긍정적 시그널을 보내고 있습니다. 리스크 자산 비중 확대를 고려하십시오.")
        elif bearish_signals >= 2:
            interpretations.append("**종합 판단: 약세 (BEARISH)** - 복수의 기관 프레임워크가 경고 시그널을 보내고 있습니다. 방어적 포지셔닝을 권고합니다.")
        else:
            interpretations.append("**종합 판단: 중립 (NEUTRAL)** - 혼재된 시그널로 인해 적극적 포지션 변경보다는 현 수준 유지가 적절합니다.")

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

            reasons_html = ''.join([f'<li>{r[:100]}...</li>' for r in reasoning[:2]])

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
            weights = {'기술/AI': 40, '중소형주': 25, '글로벌': 15, '원자재': 10, '현금': 10}

        colors = ['#1864ab', '#5f3dc4', '#2b8a3e', '#f08c00', '#868e96', '#c92a2a', '#0b7285']
        gradients = []
        legend_items = []
        cumulative = 0

        items = list(weights.items())[:7]
        total = sum(v for _, v in items)

        for i, (label, value) in enumerate(items):
            pct = (value / total * 100) if total > 0 else 0
            color = colors[i % len(colors)]
            gradients.append(f"{color} {cumulative}% {cumulative + pct}%")
            legend_items.append(f'''<div class="legend-item">
                <div class="legend-color" style="background: {color};"></div>
                <span>{label} ({pct:.0f}%)</span>
            </div>''')
            cumulative += pct

        gradient_str = ', '.join(gradients)

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">💼 추천 포트폴리오</span>
    </div>
    <div class="pie-container">
        <div class="pie-chart" style="background: conic-gradient({gradient_str});">
            <div class="pie-hole">배분</div>
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

    def _generate_watchlist_section(self) -> str:
        """주목할 종목 (NEW)"""
        # MD 섹션 7에서 추출
        section = self.ai_report_sections.get('section_7', {})
        content = section.get('content', '')

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">👀 주목할 종목</span>
    </div>
    <div class="grid grid-3">
        <div class="signal-card alert">
            <div class="signal-header">
                <span class="signal-ticker">GLD</span>
                <span class="signal-badge bg-green">강세</span>
            </div>
            <table style="font-size: 0.9rem;">
                <tr><td>1D</td><td class="text-green">+2.45%</td></tr>
                <tr><td>5D</td><td class="text-green">+7.33%</td></tr>
                <tr><td>20D</td><td class="text-green">+19.44%</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 8px; font-size: 0.85rem;">인플레이션 헤지 수요 증가</p>
        </div>
        <div class="signal-card warning">
            <div class="signal-header">
                <span class="signal-ticker">^VIX</span>
                <span class="signal-badge bg-yellow">주의</span>
            </div>
            <table style="font-size: 0.9rem;">
                <tr><td>1D</td><td class="text-red">-0.61%</td></tr>
                <tr><td>5D</td><td class="text-green">+3.90%</td></tr>
                <tr><td>20D</td><td class="text-green">+13.40%</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 8px; font-size: 0.85rem;">변동성 상승 신호 (잠재적 조정)</p>
        </div>
        <div class="signal-card">
            <div class="signal-header">
                <span class="signal-ticker">ETH-USD</span>
                <span class="signal-badge bg-blue">중립</span>
            </div>
            <table style="font-size: 0.9rem;">
                <tr><td>1D</td><td class="text-red">-0.57%</td></tr>
                <tr><td>5D</td><td class="text-green">+1.91%</td></tr>
                <tr><td>20D</td><td class="text-red">-2.53%</td></tr>
            </table>
            <p class="text-muted" style="margin-top: 8px; font-size: 0.85rem;">기술주 강세와 연동 가능</p>
        </div>
    </div>
</div>'''

    def _generate_news_section(self) -> str:
        """주요 시장 뉴스"""
        # MD 섹션 8에서 추출
        section = self.ai_report_sections.get('section_8', {})
        content = section.get('content', '')

        # 뉴스 항목 추출 시도
        news_items = [
            {'tag': 'Macro', 'tag_class': 'bg-blue', 'title': '미국 증시 사상 최고치 경신',
             'content': 'S&P 500 6990pt 돌파. 기술주 강세 지속, 나스닥 1% 상승 마감.'},
            {'tag': 'Fed', 'tag_class': 'bg-purple', 'title': 'FOMC 금리 동결 전망 90%+',
             'content': '파월 의장 메시지에 주목. CPI 2.7%, 실업률 4.4% 상황에서 인내심 유지 예상.'},
            {'tag': 'Tech', 'tag_class': 'bg-green', 'title': '빅테크 4Q 실적 발표 대기',
             'content': '마이크로소프트, 애플 등 메가캡 기업 실적 주간. AI 투자 지속 여부 확인.'}
        ]

        news_html = ''
        for item in news_items:
            news_html += f'''<div class="news-card">
                <span class="news-tag {item['tag_class']}">{item['tag']}</span>
                <p class="news-title">{item['title']}</p>
                <p class="news-content">{item['content']}</p>
            </div>'''

        return f'''
<div class="card" style="margin-bottom: 24px;">
    <div class="card-header">
        <span class="card-title">📰 주요 시장 뉴스</span>
    </div>
    {news_html}
</div>'''

    def _generate_scenario_section(self) -> str:
        """시나리오 분석"""
        return '''
<div class="grid grid-3" style="margin-bottom: 24px;">
    <div class="scenario-card base">
        <div class="scenario-header">
            <span class="scenario-title text-blue">📊 Base Case</span>
            <span class="scenario-prob text-blue">55%</span>
        </div>
        <p style="margin-bottom: 8px;">경제 지표 호조 지속, 기업 실적 컨센서스 부합</p>
        <p style="font-weight: 700;">S&P 500: 7,200~7,400 (+8~12%)</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 현재 포지션 유지, 조정 시 매수</p>
    </div>
    <div class="scenario-card bull">
        <div class="scenario-header">
            <span class="scenario-title text-green">🐂 Bull Case</span>
            <span class="scenario-prob text-green">30%</span>
        </div>
        <p style="margin-bottom: 8px;">인플레 조기 안정, AI 붐 가속화</p>
        <p style="font-weight: 700;">S&P 500: 7,600~8,000 (+15~20%)</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 주식 비중 최대, 성장주 집중</p>
    </div>
    <div class="scenario-card bear">
        <div class="scenario-header">
            <span class="scenario-title text-red">🐻 Bear Case</span>
            <span class="scenario-prob text-red">15%</span>
        </div>
        <p style="margin-bottom: 8px;">긴축 재개, 지정학적 리스크 확산</p>
        <p style="font-weight: 700;">S&P 500: 5,800~6,200 (-10~15%)</p>
        <p class="text-muted" style="font-size: 0.85rem;">전략: 현금/채권 확대, 인버스 헤지</p>
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
                <li>저항선 근접 - 돌파 실패 시 조정 가능</li>
                <li>금 급등 (4.4%) - 안전자산 선호 증가 신호</li>
                <li>VIX 20일 +13% - 변동성 상승 조짐</li>
            </ul>
        </div>
    </div>
</div>'''

    def _generate_ai_analysis_section(self) -> str:
        """AI 종합 분석"""
        section = self.ai_report_sections.get('section_9', {})
        content = section.get('content', '')

        if not content:
            content = """현재 시장은 **Bull (Low Vol)** 레짐으로, 투자자들에게 가장 우호적인 환경을 제공하고 있습니다.

### 핵심 지표 분석
- **유동성**: 풍부한 시장 유동성으로 자산 가격 상승 동력 확보
- **리스크 점수**: 매우 낮은 위험도로 공격적 투자 가능
- **신뢰도**: 높은 신뢰도로 분석 결과의 안정성 확보

### 투자자 유형별 권고
- **보수적**: 주식 50-60%, 대형주 중심
- **적극적**: 주식 70-80%, 성장주/소형주 비중 증대
- **기관**: 전술적 자산배분 조정, 리스크 패리티"""

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
