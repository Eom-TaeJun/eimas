#!/usr/bin/env python3
"""
EIMAS Report Generator
======================
종합 HTML 리포트 생성

주요 기능:
1. 일일 시장 리포트
2. 포트폴리오 성과 리포트
3. 시그널 분석 리포트
4. 리스크 대시보드

Usage:
    from lib.report_generator import ReportGenerator

    rg = ReportGenerator()
    rg.generate_daily_report()
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from lib.trading_db import TradingDB


# ============================================================================
# HTML Templates
# ============================================================================

HTML_HEADER = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
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
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        .header .date {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}

        .card h2 {{
            font-size: 1.2rem;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}

        .metric:last-child {{ border-bottom: none; }}

        .metric-label {{ color: var(--text-secondary); }}

        .metric-value {{ font-weight: 600; }}

        .metric-value.positive {{ color: var(--accent-green); }}
        .metric-value.negative {{ color: var(--accent-red); }}
        .metric-value.warning {{ color: var(--accent-yellow); }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .badge.buy {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .badge.sell {{ background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }}
        .badge.hold {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}

        .progress-bar .fill {{
            height: 100%;
            border-radius: 4px;
        }}

        .progress-bar .fill.green {{ background: var(--accent-green); }}
        .progress-bar .fill.red {{ background: var(--accent-red); }}
        .progress-bar .fill.yellow {{ background: var(--accent-yellow); }}
        .progress-bar .fill.blue {{ background: var(--accent-blue); }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .section {{
            margin-bottom: 30px;
        }}

        .section-title {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent-blue);
        }}

        .footer {{
            text-align: center;
            padding: 30px 0;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            margin-top: 30px;
        }}

        .allocation-bar {{
            display: flex;
            height: 30px;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .allocation-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
<div class="container">
"""

HTML_FOOTER = """
    <div class="footer">
        <p>Generated by EIMAS (Economic Intelligence Multi-Agent System)</p>
        <p>{timestamp}</p>
    </div>
</div>
</body>
</html>
"""


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """리포트 생성기"""

    def __init__(self, db: TradingDB = None, output_dir: str = None):
        self.db = db or TradingDB()
        self.output_dir = Path(output_dir or '/home/tj/projects/autoai/eimas/outputs')
        self.output_dir.mkdir(exist_ok=True)

    def _get_color_class(self, value: float, thresholds: tuple = (0, 0)) -> str:
        """값에 따른 색상 클래스"""
        if value > thresholds[1]:
            return "positive"
        elif value < thresholds[0]:
            return "negative"
        return ""

    def _generate_header(self, title: str, subtitle: str = "") -> str:
        """헤더 HTML 생성"""
        return f"""
        <div class="header">
            <h1>📊 {title}</h1>
            <p class="date">{subtitle or date.today().isoformat()}</p>
        </div>
        """

    def generate_daily_report(self) -> str:
        """일일 리포트 생성"""
        # 데이터 수집
        signals = self.db.get_recent_signals(hours=24)
        portfolios = self.db.get_portfolios(limit=4)
        session_stats = self.db.get_session_stats("SPY", days=30)
        db_summary = self.db.get_summary()

        # HTML 생성
        html = HTML_HEADER.format(title="EIMAS Daily Report")
        html += self._generate_header("Daily Market Report", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Summary Cards
        html += '<div class="grid">'

        # Signal Summary
        signal_count = len(signals)
        buy_signals = sum(1 for s in signals if s.get('signal_action') == 'buy')
        sell_signals = sum(1 for s in signals if s.get('signal_action') == 'sell')

        html += f"""
        <div class="card">
            <h2>📡 Today's Signals</h2>
            <div class="metric">
                <span class="metric-label">Total Signals</span>
                <span class="metric-value">{signal_count}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Buy Signals</span>
                <span class="metric-value positive">{buy_signals}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sell Signals</span>
                <span class="metric-value negative">{sell_signals}</span>
            </div>
        </div>
        """

        # Database Status
        html += f"""
        <div class="card">
            <h2>📁 Database Status</h2>
            <div class="metric">
                <span class="metric-label">Total Signals</span>
                <span class="metric-value">{db_summary['total_signals']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Portfolios</span>
                <span class="metric-value">{db_summary['total_portfolios']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Executions</span>
                <span class="metric-value">{db_summary['total_executions']}</span>
            </div>
        </div>
        """

        # Session Stats
        if session_stats:
            best_session = max(session_stats.items(), key=lambda x: x[1])
            worst_session = min(session_stats.items(), key=lambda x: x[1])

            html += f"""
            <div class="card">
                <h2>⏰ Session Analysis (30d avg)</h2>
                <div class="metric">
                    <span class="metric-label">Best Session</span>
                    <span class="metric-value positive">{best_session[0]} ({best_session[1]:+.2%})</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Worst Session</span>
                    <span class="metric-value negative">{worst_session[0]} ({worst_session[1]:+.2%})</span>
                </div>
            </div>
            """

        html += '</div>'  # End grid

        # Portfolios Section
        if portfolios:
            html += '<div class="section"><h2 class="section-title">💼 Portfolio Recommendations</h2>'
            html += '<div class="grid">'

            colors = ['#3fb950', '#58a6ff', '#d29922', '#f85149', '#8b949e']

            for p in portfolios:
                allocations = p.get('allocations', {})
                profile = p.get('profile_type', 'Unknown').upper()
                sharpe = p.get('expected_sharpe', 0)
                exp_return = p.get('expected_return', 0) * 100
                exp_risk = p.get('expected_risk', 0) * 100

                html += f"""
                <div class="card">
                    <h2>{profile}</h2>
                    <div class="metric">
                        <span class="metric-label">Expected Return</span>
                        <span class="metric-value positive">{exp_return:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Expected Risk</span>
                        <span class="metric-value">{exp_risk:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value">{sharpe:.2f}</span>
                    </div>
                    <div class="allocation-bar">
                """

                for i, (ticker, weight) in enumerate(sorted(allocations.items(), key=lambda x: -x[1])):
                    color = colors[i % len(colors)]
                    width = weight * 100
                    if width > 5:
                        html += f'<div class="allocation-segment" style="width:{width}%;background:{color}">{ticker}</div>'

                html += '</div></div>'

            html += '</div></div>'

        # Signals Table
        if signals:
            html += '<div class="section"><h2 class="section-title">📡 Recent Signals</h2>'
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Source</th>
                        <th>Action</th>
                        <th>Ticker</th>
                        <th>Conviction</th>
                    </tr>
                </thead>
                <tbody>
            """

            for s in signals[:10]:
                action = s.get('signal_action', 'hold')
                badge_class = action if action in ['buy', 'sell', 'hold'] else 'hold'
                conviction = s.get('conviction', 0) * 100

                html += f"""
                <tr>
                    <td>{s.get('timestamp', '')[:16]}</td>
                    <td>{s.get('signal_source', '')}</td>
                    <td><span class="badge {badge_class}">{action.upper()}</span></td>
                    <td>{s.get('ticker', 'SPY')}</td>
                    <td>{conviction:.0f}%</td>
                </tr>
                """

            html += '</tbody></table></div>'

        html += HTML_FOOTER.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Save
        filename = f"daily_report_{date.today().isoformat()}.html"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✅ Report saved: {filepath}")
        return str(filepath)

    def generate_risk_report(
        self,
        holdings: Dict[str, float],
        risk_data: Dict[str, Any]
    ) -> str:
        """리스크 리포트 생성"""
        html = HTML_HEADER.format(title="EIMAS Risk Report")
        html += self._generate_header("Portfolio Risk Analysis")

        # Risk Metrics
        html += '<div class="grid">'

        risk_level = risk_data.get('risk_level', 'medium')
        level_color = {
            'low': 'positive',
            'medium': 'warning',
            'high': 'negative',
            'extreme': 'negative'
        }.get(risk_level, '')

        html += f"""
        <div class="card">
            <h2>⚠️ Risk Level</h2>
            <div class="metric">
                <span class="metric-label">Overall Risk</span>
                <span class="metric-value {level_color}">{risk_level.upper()}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Annual Volatility</span>
                <span class="metric-value">{risk_data.get('annual_vol', 0):.1f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value negative">{risk_data.get('max_drawdown', 0):.1f}%</span>
            </div>
        </div>
        """

        html += f"""
        <div class="card">
            <h2>📉 Value at Risk (1-day)</h2>
            <div class="metric">
                <span class="metric-label">VaR 95%</span>
                <span class="metric-value">${risk_data.get('var_95', 0):,.0f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">VaR 99%</span>
                <span class="metric-value">${risk_data.get('var_99', 0):,.0f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">CVaR 95%</span>
                <span class="metric-value">${risk_data.get('cvar_95', 0):,.0f}</span>
            </div>
        </div>
        """

        html += f"""
        <div class="card">
            <h2>📊 Market Metrics</h2>
            <div class="metric">
                <span class="metric-label">Beta (vs SPY)</span>
                <span class="metric-value">{risk_data.get('beta', 1.0):.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Correlation</span>
                <span class="metric-value">{risk_data.get('correlation_to_spy', 0):.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe (Est)</span>
                <span class="metric-value">{risk_data.get('sharpe_estimate', 0):.2f}</span>
            </div>
        </div>
        """

        html += '</div>'

        # Holdings
        html += '<div class="section"><h2 class="section-title">💼 Current Holdings</h2>'
        html += '<table><thead><tr><th>Asset</th><th>Weight</th><th>Allocation</th></tr></thead><tbody>'

        colors = ['#3fb950', '#58a6ff', '#d29922', '#f85149', '#8b949e']
        for i, (ticker, weight) in enumerate(sorted(holdings.items(), key=lambda x: -x[1])):
            color = colors[i % len(colors)]
            html += f"""
            <tr>
                <td>{ticker}</td>
                <td>{weight:.1%}</td>
                <td>
                    <div class="progress-bar">
                        <div class="fill" style="width:{weight*100}%;background:{color}"></div>
                    </div>
                </td>
            </tr>
            """

        html += '</tbody></table></div>'

        html += HTML_FOOTER.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Save
        filename = f"risk_report_{date.today().isoformat()}.html"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✅ Risk report saved: {filepath}")
        return str(filepath)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Report Generator Test")
    print("=" * 60)

    rg = ReportGenerator()

    # Daily Report
    print("\nGenerating daily report...")
    daily_path = rg.generate_daily_report()

    # Risk Report
    print("\nGenerating risk report...")
    holdings = {"SPY": 0.40, "QQQ": 0.20, "TLT": 0.25, "GLD": 0.10, "CASH": 0.05}
    risk_data = {
        'risk_level': 'medium',
        'annual_vol': 13.7,
        'max_drawdown': 9.2,
        'var_95': 1006,
        'var_99': 2992,
        'cvar_95': 1854,
        'beta': 0.65,
        'correlation_to_spy': 0.96,
        'sharpe_estimate': 1.67,
    }
    risk_path = rg.generate_risk_report(holdings, risk_data)

    print("\n" + "=" * 60)
    print("Reports Generated!")
    print(f"  Daily: {daily_path}")
    print(f"  Risk:  {risk_path}")
    print("=" * 60)
