"""
EIMAS Light Theme CSS
=====================
Clean professional light mode CSS theme for HTML reports.

Extracted from lib.final_report_agent for better modularity.
"""

CSS_LIGHT_THEME = """
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
